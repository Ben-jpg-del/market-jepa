"""Future-block JEPA pilot, separate from the original F-JEPA implementation.

Availability is a model input. A missing stock is also excluded as an attention
key. Teacher targets see only the supplied future block, never the old context.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .features import (FEATURE_SHORTCUT_PARTNERS, STOCK_FAMILIES, STOCK_FEATURE_GROUPS, STOCK_FEATURES,
                       stock_families_for, stock_feature_names)


@dataclass
class PilotConfig:
    stock_features: int = 14
    market_features: int = 8
    context_days: int = 60
    patch_days: int = 5
    width: int = 128
    temporal_layers: int = 3
    cross_stock_layers: int = 1
    heads: int = 4
    predictor_width: int = 64
    predictor_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.0
    sigreg_directions: int = 256
    sigreg_points: int = 17
    sigreg_weight: float = 0.05
    # "bundled": one token per five-day patch holding all indicators (original design).
    # "family": one token stream per indicator family (fast/trend/risk/book); the
    # stock vector is the concatenation of per-family slots of family_width each.
    # "feature": one token stream per indicator (14 streams), slot width = width // 14, zero-padded.
    tokenization: str = "bundled"
    family_width: int = 32
    # "block": predict the teacher embedding of a hidden/future block (original).
    # "family_mask": hide one indicator family for a stock and predict the teacher's
    # summary of that family from the others (T-JEPA column masking analogue).
    # "feature_mask": hide one indicator plus its arithmetic shortcut partners, predict the
    # teacher's summary of that indicator (requires tokenization="feature").
    # "future_family" (current): predict the teacher's summary of each family on the window ending at t+h,
    # for h in `horizons`, with a horizon-conditioned query; SIGReg in a projection head; auxiliary rank head.
    objective: str = "block"
    feature_layout: str = "v1"      # "v1" (14/21/28-channel layouts) or "v2" (15 stock / 11 market, features.STOCK_FEATURES_ASSEMBLED)
    horizons: list = (5, 20)        # future_family only: sessions ahead for the predicted family summaries (stored as a list: JSON-stable)
    aux_weight: float = 0.1         # future_family only: weight of the auxiliary rank-head loss inside the prediction term
    prediction_weight: float = 1.0  # future_family only: weight of the JEPA prediction term; 0 = supervised encoder (aux head only)
    aux_target: str = "return_rank" # future_family only: "return_rank" (per-date normal score of the SPY-relative return)
                                    # or "vol_rank" (per-date normal score of the realised volatility over the next h sessions)
    aux_loss: str = "mse"           # future_family only: "mse" on the target, or "sharpe" = -Sharpe over batch dates of the
                                    # dollar-neutral portfolio whose weights are the demeaned head outputs (needs aux_raw)

    def __post_init__(self):
        self.horizons = [int(h) for h in self.horizons]  # list, so asdict() == json round trip (snapshot/metadata equality checks)
        if self.feature_layout not in {"v1", "v2"}:
            raise ValueError("feature_layout must be the layout tag v1 (14 stock channels) or v2 (15 stock / 11 market channels)")
        if self.context_days % self.patch_days:
            raise ValueError("context_days must be divisible by patch_days")
        if self.width % self.heads or self.predictor_width % self.heads:
            raise ValueError("Encoder/predictor widths must be divisible by heads")
        if not 0 <= self.sigreg_weight <= 1:
            raise ValueError("sigreg_weight must be in [0, 1]")
        if self.tokenization not in {"bundled", "family", "feature"}:
            raise ValueError("tokenization must be bundled, family, or feature")
        if self.objective not in {"block", "family_mask", "feature_mask", "future_family"}:
            raise ValueError("objective must be block, family_mask, feature_mask, or future_family")
        if self.objective == "future_family":
            if self.tokenization != "family":
                raise ValueError("future_family objective requires family tokenization")
            if not self.horizons or any(h < 1 for h in self.horizons):
                raise ValueError("horizons must be positive")
            families = stock_families_for(self.stock_features, self.feature_layout)
            if self.width != self.family_width * len(families):
                raise ValueError("future_family requires width == family_width x families (fixed slot width, no padding)")
            if not 0 <= self.aux_weight:
                raise ValueError("aux_weight must be non-negative")
            if not 0 <= self.prediction_weight:
                raise ValueError("prediction_weight must be non-negative")
            if self.prediction_weight == 0 and self.aux_weight == 0:
                raise ValueError("future_family needs a prediction term or an aux term")
            if self.aux_target not in {"return_rank", "vol_rank"}:
                raise ValueError("aux_target must be return_rank or vol_rank")
            if self.aux_loss not in {"mse", "sharpe"}:
                raise ValueError("aux_loss must be mse or sharpe")
            if self.aux_loss == "sharpe" and self.aux_target != "return_rank":
                raise ValueError("the sharpe aux loss is defined on returns (aux_target=return_rank)")
        if self.objective == "family_mask" and self.tokenization != "family":
            raise ValueError("family_mask objective requires family tokenization")
        if self.objective == "feature_mask" and self.tokenization != "feature":
            raise ValueError("feature_mask objective requires feature tokenization")
        if self.tokenization == "family":
            families = stock_families_for(self.stock_features, self.feature_layout)  # raises for unsupported widths
            if sum(len(idx) for _, idx in families) != self.stock_features:
                raise ValueError("Stock families must cover every stock feature")
            if self.width // len(families) < 4:
                raise ValueError("family tokenization needs width >= 4 x families")
        if self.tokenization == "feature" and self.width // self.stock_features < 4:
            raise ValueError("feature tokenization needs width >= 4 x stock_features")

    @property
    def token_groups(self):
        """(name, channel indices) per token stream for grouped tokenizations."""
        if self.tokenization == "family":
            return stock_families_for(self.stock_features, self.feature_layout)
        if self.tokenization == "feature":
            names = stock_feature_names(self.stock_features, self.feature_layout)
            return tuple((name, (i,)) for i, name in enumerate(names))
        return ()

    @property
    def slot_width(self) -> int:
        # Equal slots per token group; any remainder of `width` is zero padding.
        return self.width // max(1, len(self.token_groups))


def _transformer(width, layers, cfg):
    layer = nn.TransformerEncoderLayer(
        width, cfg.heads, width * cfg.ff_multiplier, cfg.dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    stack = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    # TransformerEncoder clones one initialized layer; initialize clones apart.
    for module in stack.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
    return stack


def _patch_tokens(values: Tensor, observed: Tensor, projection, config, position) -> tuple[Tensor, Tensor]:
    """Mask BEFORE projection, so unavailable NaNs or arbitrary filler vanish."""
    if values.shape != observed.shape:
        raise ValueError("Feature values and observation masks must have identical shapes")
    observed = observed.bool()
    if torch.any(observed & ~torch.isfinite(values)):
        raise ValueError("Observed feature values must be finite")
    n, days, features = values.shape
    if days % config.patch_days or days > config.context_days:
        raise ValueError("History must contain complete patches within context_days")
    patches = days // config.patch_days
    clean = torch.where(observed, values, 0.0)
    joined = torch.cat((clean, observed.to(values.dtype)), dim=-1)
    projected = projection(joined.reshape(n, patches, -1))
    valid = observed.reshape(n, patches, -1).any(-1)
    return projected + position[-patches:], valid


def _last_valid_summary(encoded: Tensor, valid: Tensor) -> Tensor:
    positions = torch.arange(valid.shape[1], device=valid.device)[None, :]
    last = torch.where(valid, positions, -1).amax(-1).clamp_min(0)
    summary = encoded[torch.arange(encoded.shape[0], device=encoded.device), last]
    return summary * valid.any(-1, keepdim=True)


class FamilyEncoder(nn.Module):
    """Per-family token streams; stock vector = concatenated per-family slots.

    Each indicator family gets its own input projection and learned family
    embedding, runs through the shared temporal transformer, and is summarised
    by its last valid patch. Summaries are compressed to ``family_width`` and
    concatenated, so every family keeps its own residual slot inside the
    ``width``-dimensional stock vector that cross-stock attention then refines.
    Uncompressed family summaries are also returned as targets for the
    family-mask objective.
    """

    def __init__(self, config: PilotConfig):
        super().__init__()
        self.config = config
        c = config
        groups = c.token_groups
        self.family_names = [name for name, _ in groups]
        self.family_index = [list(idx) for _, idx in groups]
        self.family_projections = nn.ModuleList(
            [nn.Linear(c.patch_days * len(idx) * 2, c.width) for idx in self.family_index])
        self.family_embedding = nn.Parameter(torch.randn(len(self.family_index), c.width) * 0.02)
        self.family_compress = nn.ModuleList([nn.Linear(c.width, c.slot_width) for _ in self.family_index])
        self.slot_padding = c.width - c.slot_width * len(self.family_index)  # zero pad when width % groups != 0
        self.market_projection = nn.Linear(c.patch_days * c.market_features * 2, c.width)
        self.position = nn.Parameter(torch.randn(c.context_days // c.patch_days, c.width) * 0.02)
        self.register = nn.Parameter(torch.randn(1, 1, c.width) * 0.02)
        self.temporal = _transformer(c.width, c.temporal_layers, c)
        self.cross_stock = _transformer(c.width, c.cross_stock_layers, c)
        self.output = nn.Linear(c.width, c.width)

    def forward(self, stock, market, stock_observed, market_observed, stock_valid):
        if stock.ndim != 4 or market.ndim != 3:
            raise ValueError("Expected stock[B,K,T,C] and market[B,T,C]")
        b, k, days, features = stock.shape
        if market.shape[:2] != (b, days) or stock_valid.shape != (b, k):
            raise ValueError("Stock, market, and eligibility axes are inconsistent")
        stock_observed = stock_observed.bool() & stock_valid.bool()[..., None, None]
        n = b * k
        flat = stock.reshape(n, days, features)
        flat_obs = stock_observed.reshape(n, days, features)
        tokens, valids = [], []
        for f, (idx, projection) in enumerate(zip(self.family_index, self.family_projections)):
            t, v = _patch_tokens(flat[..., idx], flat_obs[..., idx], projection, self.config, self.position)
            tokens.append(t + self.family_embedding[f])
            valids.append(v)
        m, mv = _patch_tokens(market, market_observed, self.market_projection, self.config, self.position)
        all_tokens = torch.cat(tokens + [m], dim=0)
        valid = torch.cat(valids + [mv], dim=0)
        safe_valid = valid.clone()
        safe_valid[~valid.any(-1), -1] = True
        encoded = self.temporal(all_tokens, src_key_padding_mask=~safe_valid)
        summary = _last_valid_summary(encoded, valid)
        count = len(self.family_index)
        family_summaries = torch.stack([summary[f * n:(f + 1) * n] for f in range(count)], dim=1)  # (n, F, width)
        family_valid = torch.stack([v.any(-1) for v in valids], dim=1)  # (n, F)
        slots = [self.family_compress[f](family_summaries[:, f]) * family_valid[:, f, None] for f in range(count)]
        if self.slot_padding:
            slots.append(torch.zeros(n, self.slot_padding, device=stock.device, dtype=slots[0].dtype))
        stock_tokens = torch.cat(slots, dim=-1).reshape(b, k, -1)
        market_summary = summary[count * n:]
        stock_ok = family_valid.any(-1).reshape(b, k) & stock_valid.bool()
        market_ok = mv.any(-1)
        entities = torch.cat((stock_tokens, market_summary[:, None], self.register.expand(b, -1, -1)), dim=1)
        entity_valid = torch.cat((stock_ok, market_ok[:, None],
                                  torch.ones(b, 1, device=stock.device, dtype=torch.bool)), dim=1)
        out = self.output(self.cross_stock(entities, src_key_padding_mask=~entity_valid))
        return {
            "stock_embeddings": out[:, :k] * stock_ok[..., None],
            "market_embedding": out[:, k] * market_ok[:, None],
            "valid_stock_mask": stock_ok,
            "family_summaries": family_summaries.reshape(b, k, count, -1),
            "family_valid": family_valid.reshape(b, k, count) & stock_valid.bool()[..., None],
        }


class Encoder(nn.Module):
    def __init__(self, config: PilotConfig):
        super().__init__()
        self.config = config
        c = config
        self.stock_projection = nn.Linear(c.patch_days * c.stock_features * 2, c.width)
        self.market_projection = nn.Linear(c.patch_days * c.market_features * 2, c.width)
        self.position = nn.Parameter(torch.randn(c.context_days // c.patch_days, c.width) * 0.02)
        self.register = nn.Parameter(torch.randn(1, 1, c.width) * 0.02)
        self.temporal = _transformer(c.width, c.temporal_layers, c)
        self.cross_stock = _transformer(c.width, c.cross_stock_layers, c)
        self.output = nn.Linear(c.width, c.width)

    def _patch(self, values: Tensor, observed: Tensor, projection):
        # Mask BEFORE projection, so unavailable NaNs or arbitrary filler vanish.
        if values.shape != observed.shape:
            raise ValueError("Feature values and observation masks must have identical shapes")
        observed = observed.bool()
        if torch.any(observed & ~torch.isfinite(values)):
            raise ValueError("Observed feature values must be finite")
        n, days, features = values.shape
        if days % self.config.patch_days or days > self.config.context_days:
            raise ValueError("History must contain complete patches within context_days")
        patches = days // self.config.patch_days
        clean = torch.where(observed, values, 0.0)
        joined = torch.cat((clean, observed.to(values.dtype)), dim=-1)
        projected = projection(joined.reshape(n, patches, -1))
        valid = observed.reshape(n, patches, -1).any(-1)
        return projected + self.position[-patches:], valid

    def forward(self, stock, market, stock_observed, market_observed, stock_valid):
        if stock.ndim != 4 or market.ndim != 3:
            raise ValueError("Expected stock[B,K,T,C] and market[B,T,C]")
        b, k, days, features = stock.shape
        if market.shape[:2] != (b, days) or stock_valid.shape != (b, k):
            raise ValueError("Stock, market, and eligibility axes are inconsistent")
        # Entirely ineligible histories must not affect any other stock.
        stock_observed = stock_observed.bool() & stock_valid.bool()[..., None, None]
        s, sv = self._patch(stock.reshape(b * k, days, features),
                            stock_observed.reshape(b * k, days, features), self.stock_projection)
        m, mv = self._patch(market, market_observed, self.market_projection)
        tokens = torch.cat((s, m), dim=0)
        valid = torch.cat((sv, mv), dim=0)
        safe_valid = valid.clone()
        safe_valid[~valid.any(-1), -1] = True  # avoid all-masked attention rows
        encoded = self.temporal(tokens, src_key_padding_mask=~safe_valid)
        positions = torch.arange(valid.shape[1], device=valid.device)[None, :]
        last = torch.where(valid, positions, -1).amax(-1).clamp_min(0)
        summary = encoded[torch.arange(encoded.shape[0], device=encoded.device), last]
        summary = summary * valid.any(-1, keepdim=True)
        stock_ok = valid[:b * k].any(-1).reshape(b, k) & stock_valid.bool()
        market_ok = valid[b * k:].any(-1)
        entities = torch.cat((summary[:b * k].reshape(b, k, -1),
                              summary[b * k:, None], self.register.expand(b, -1, -1)), dim=1)
        entity_valid = torch.cat((stock_ok, market_ok[:, None],
                                  torch.ones(b, 1, device=stock.device, dtype=torch.bool)), dim=1)
        out = self.output(self.cross_stock(entities, src_key_padding_mask=~entity_valid))
        return {
            "stock_embeddings": out[:, :k] * stock_ok[..., None],
            "market_embedding": out[:, k] * market_ok[:, None],
            "valid_stock_mask": stock_ok,
        }


class Predictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.projection = nn.Linear(c.width, c.predictor_width)
        self.layers = _transformer(c.predictor_width, c.predictor_layers, c)
        self.output = nn.Linear(c.predictor_width, c.width)

    def forward(self, encoded, query: Tensor | None = None):
        stock, market = encoded["stock_embeddings"], encoded["market_embedding"]
        if query is not None:
            stock = stock + query
        b, k, _ = stock.shape
        values = self.projection(torch.cat((stock, market[:, None]), dim=1))
        # Market token remains a safe key even if no market observation exists.
        valid = torch.cat((encoded["valid_stock_mask"],
                           torch.ones(b, 1, device=stock.device, dtype=torch.bool)), dim=1)
        return self.output(self.layers(values, src_key_padding_mask=~valid))[:, :k]


def sigreg(x: Tensor, directions=256, points=17) -> Tensor:
    """LeJEPA Algorithm 1 EP statistic, averaged over fresh unit directions.

    The N multiplier is the paper's test-statistic scaling. No centering,
    projection standardization, or per-vector normalization is performed.
    """
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("SIGReg needs a nonempty [samples, dimensions] matrix")
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        a = torch.randn(x.shape[1], directions, device=x.device, dtype=torch.float32)
        a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
        t = torch.linspace(-5, 5, points, device=x.device)
        gaussian = torch.exp(-0.5 * t.square())
        phase = (x @ a).unsqueeze(-1) * t
        real, imaginary = phase.cos().mean(0), phase.sin().mean(0)
        error = ((real - gaussian).square() + imaginary.square()) * gaussian
        return torch.trapezoid(error, t, dim=-1).mean() * x.shape[0]


def _get(batch: Mapping, name, alias=None):
    return batch[name] if name in batch else batch[alias or name]


class FutureJEPA(nn.Module):
    def __init__(self, config: PilotConfig | None = None):
        super().__init__()
        self.config = config or PilotConfig()
        self.online = FamilyEncoder(self.config) if self.config.tokenization in ("family", "feature") else Encoder(self.config)
        self.teacher = copy.deepcopy(self.online).requires_grad_(False).eval()
        self.predictor = Predictor(self.config)
        if self.config.objective in ("family_mask", "feature_mask"):
            groups = self.config.token_groups
            self.family_query = nn.Parameter(torch.randn(len(groups), self.config.width) * 0.02)
            channels = torch.zeros(len(groups), self.config.stock_features, dtype=torch.bool)
            hidden = torch.zeros(len(groups), self.config.stock_features, dtype=torch.bool)
            name_to_index = {name: i for i, name in enumerate(stock_feature_names(self.config.stock_features))}
            for f, (name, idx) in enumerate(groups):
                channels[f, list(idx)] = True
                hidden[f, list(idx)] = True
                if self.config.objective == "feature_mask":
                    for partner in FEATURE_SHORTCUT_PARTNERS.get(name, ()):
                        hidden[f, name_to_index[partner]] = True
            self.register_buffer("family_channels", channels, persistent=False)  # target channels per group
            self.register_buffer("hide_channels", hidden, persistent=False)      # channels hidden when that group is the target
        if self.config.objective == "future_family":
            groups = self.config.token_groups
            self.family_query = nn.Parameter(torch.randn(len(groups), self.config.width) * 0.02)
            self.horizon_embedding = nn.Parameter(torch.randn(len(self.config.horizons), self.config.width) * 0.02)
            self.projection_head = nn.Linear(self.config.width, self.config.width)   # SIGReg is applied here, not on the readout embedding
            self.aux_head = nn.Linear(self.config.width, len(self.config.horizons))  # per-date rank of the h-day relative return

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def encode(self, stock, market, stock_obs, market_obs, eligibility):
        return self.online(stock, market, stock_obs, market_obs, eligibility)

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        online = dict(self.online.named_parameters())
        for name, target in self.teacher.named_parameters():
            target.lerp_(online[name], 1 - momentum)
        online_buffers = dict(self.online.named_buffers())
        for name, target in self.teacher.named_buffers():
            target.copy_(online_buffers[name])

    def loss(self, batch: Mapping, context_mask_probability=0.0):
        if self.config.objective == "future_family":
            if context_mask_probability:
                raise ValueError("future_family does not combine with context masking")
            return self._future_family_loss(batch)
        if self.config.objective in ("family_mask", "feature_mask"):
            if context_mask_probability:
                raise ValueError("group-mask objectives do not combine with context masking")
            return self._family_mask_loss(batch)
        c = self.config
        stock_observed = _get(batch, "stock_observed", "stock_obs").bool()
        eligible = _get(batch, "stock_valid", "eligibility").bool()
        if context_mask_probability:
            stock_observed = stock_observed.clone()
            b, k, days, _ = stock_observed.shape
            patches = days // c.patch_days
            if patches < 3:
                raise ValueError("Context masking requires at least three patches")
            for row in range(b):
                if torch.rand((), device=stock_observed.device) < context_mask_probability:
                    candidates = torch.where(eligible[row])[0]
                    count = max(1, int(0.25 * len(candidates))) if len(candidates) else 0
                    chosen = candidates[torch.randperm(len(candidates), device=candidates.device)[:count]]
                    start = torch.randint(patches - 2, (), device=stock_observed.device).item()
                    stock_observed[row, chosen, start*c.patch_days:(start+2)*c.patch_days] = False
        context = self.encode(batch["stock"], batch["market"], stock_observed,
                              _get(batch, "market_observed", "market_obs"), eligible)
        target_valid = batch["target_valid"].bool() & context["valid_stock_mask"]
        future_args = (batch["future_stock"], batch["future_market"],
                       _get(batch, "future_stock_observed", "future_stock_obs"),
                       _get(batch, "future_market_observed", "future_market_obs"), target_valid)
        with torch.no_grad():
            target = self.teacher(*future_args)
        future_online = self.online(*future_args)
        valid = target_valid & target["valid_stock_mask"]
        prediction_valid = valid & batch.get("prediction_valid", valid).bool()
        if not prediction_valid.any():
            raise ValueError("Batch has no valid future targets")
        pred = self.predictor(context)
        mse = (pred[prediction_valid].float() - target["stock_embeddings"][prediction_valid].float()).square().mean()
        regularizer = 0.5 * (
            sigreg(context["stock_embeddings"][context["valid_stock_mask"]], c.sigreg_directions, c.sigreg_points)
            + sigreg(future_online["stock_embeddings"][valid], c.sigreg_directions, c.sigreg_points)
        )
        return {"loss": (1-c.sigreg_weight)*mse + c.sigreg_weight*regularizer,
                "prediction_loss": mse, "sigreg_loss": regularizer,
                "context_embeddings": context["stock_embeddings"].detach(),
                "valid_stock_mask": context["valid_stock_mask"].detach()}


    def _family_mask_loss(self, batch: Mapping):
        """Hide one observed indicator family per eligible stock; predict its teacher summary."""
        c = self.config
        stock = batch["stock"]
        stock_observed = _get(batch, "stock_observed", "stock_obs").bool()
        eligible = _get(batch, "stock_valid", "eligibility").bool()
        market, market_observed = batch["market"], _get(batch, "market_observed", "market_obs")
        b, k, days, features = stock.shape
        count = self.family_channels.shape[0]
        # Which families have any observed value for each stock (b, k, F).
        observed_by_family = torch.einsum("bktc,fc->bkf", stock_observed.float(), self.family_channels.float()) > 0
        # Eligible targets: observed, and hiding the target with its shortcut partners must
        # leave at least one observed channel outside the hidden set as context.
        observed_channels = stock_observed.any(2)  # (b, k, C)
        remaining = torch.einsum("bkc,fc->bkf", observed_channels.float(), (~self.hide_channels).float()) > 0
        candidates = observed_by_family & remaining
        scores = torch.rand(b, k, count, device=stock.device).masked_fill(~candidates, -1.0)
        chosen = scores.argmax(-1)  # (b, k)
        has_choice = candidates.any(-1) & eligible
        hidden_channels = self.hide_channels[chosen]  # (b, k, C)
        masked_observed = stock_observed & ~hidden_channels[:, :, None, :]
        context = self.encode(stock, market, masked_observed, market_observed, eligible)
        with torch.no_grad():
            teacher = self.teacher(stock, market, stock_observed, market_observed, eligible)
        full_online = self.online(stock, market, stock_observed, market_observed, eligible)
        target = torch.gather(teacher["family_summaries"], 2,
                              chosen[:, :, None, None].expand(-1, -1, 1, c.width)).squeeze(2)
        target_valid = torch.gather(teacher["family_valid"], 2, chosen[:, :, None]).squeeze(2)
        valid = has_choice & context["valid_stock_mask"] & target_valid
        if not valid.any():
            raise ValueError("Batch has no valid family targets")
        pred = self.predictor(context, query=self.family_query[chosen])
        mse = (pred[valid].float() - target[valid].float()).square().mean()
        regularizer = 0.5 * (
            sigreg(context["stock_embeddings"][context["valid_stock_mask"]], c.sigreg_directions, c.sigreg_points)
            + sigreg(full_online["stock_embeddings"][full_online["valid_stock_mask"]], c.sigreg_directions, c.sigreg_points))
        return {"loss": (1-c.sigreg_weight)*mse + c.sigreg_weight*regularizer,
                "prediction_loss": mse, "sigreg_loss": regularizer,
                "context_embeddings": full_online["stock_embeddings"].detach(),
                "valid_stock_mask": full_online["valid_stock_mask"].detach()}

    def _future_family_loss(self, batch: Mapping):
        """for each horizon h and family f, predict the teacher's summary of family f on the window ending
        at t+h from the context embedding, with query = family_query[f] + horizon_embedding[h]."""
        c = self.config
        stock_observed = _get(batch, "stock_observed", "stock_obs").bool()
        eligible = _get(batch, "stock_valid", "eligibility").bool()
        context = self.online(batch["stock"], batch["market"], stock_observed, _get(batch, "market_observed", "market_obs"), eligible)
        valid_context = context["valid_stock_mask"]
        per_h, per_pair = {}, []
        future_valid = batch["future_valid"].bool() if c.prediction_weight > 0 else None   # [B,K,H]
        for hi, h in enumerate(c.horizons if c.prediction_weight > 0 else []):
            f_args = (batch["future_stock"][:, :, hi], batch["future_market"][:, hi],
                      _get(batch, "future_stock_observed", "future_stock_obs")[:, :, hi].bool(),
                      _get(batch, "future_market_observed", "future_market_obs")[:, hi].bool(),
                      eligible & future_valid[:, :, hi])
            with torch.no_grad():
                target = self.teacher(*f_args)
            losses = []
            for f in range(len(c.token_groups)):
                mask = valid_context & future_valid[:, :, hi] & target["valid_stock_mask"] & target["family_valid"][:, :, f]
                if not mask.any():
                    continue
                pred = self.predictor(context, query=self.family_query[f] + self.horizon_embedding[hi])
                err = (pred[mask].float() - target["family_summaries"][:, :, f][mask].float()).square().mean()
                losses.append(err); per_pair.append(err)
            per_h[h] = torch.stack(losses).mean() if losses else None
        if c.prediction_weight > 0 and not per_pair:
            raise ValueError("Batch has no valid future family targets")
        prediction = torch.stack(per_pair).mean() if per_pair else context["stock_embeddings"].new_zeros(())
        aux_pred = self.aux_head(context["stock_embeddings"])          # [B,K,H]
        aux_mask = batch["aux_valid"].bool() & valid_context[..., None]
        if c.aux_loss == "sharpe":
            aux = self._sharpe_loss(aux_pred, batch["aux_raw"], aux_mask)
        else:
            aux = ((aux_pred[aux_mask].float() - batch["aux_target"][aux_mask].float()).square().mean()
                   if aux_mask.any() else prediction.new_zeros(()))
        regularizer = sigreg(self.projection_head(context["stock_embeddings"])[valid_context], c.sigreg_directions, c.sigreg_points)
        total = (1 - c.sigreg_weight) * (c.prediction_weight * prediction + c.aux_weight * aux) + c.sigreg_weight * regularizer
        out = {"loss": total, "prediction_loss": prediction, "sigreg_loss": regularizer, "aux_loss": aux,
               "context_embeddings": context["stock_embeddings"].detach(), "valid_stock_mask": valid_context.detach()}
        for h, value in per_h.items():
            out[f"prediction_loss_h{h}"] = value.detach() if value is not None else prediction.new_zeros(())
        return out


    @staticmethod
    def _sharpe_loss(scores: Tensor, raw_returns: Tensor, mask: Tensor, min_stocks: int = 4) -> Tensor:
        """-Sharpe (over the batch's dates) of the dollar-neutral, unit-gross portfolio whose weights are the demeaned
        head outputs, per horizon, averaged over horizons. scores/raw_returns/mask are [B,K,H]."""
        per_h = []
        for hi in range(scores.shape[-1]):
            rets = []
            for b in range(scores.shape[0]):
                m = mask[b, :, hi]
                if int(m.sum()) < min_stocks:
                    continue
                s = scores[b, :, hi][m].float(); s = s - s.mean()
                w = s / (s.abs().sum() + 1e-8)
                rets.append((w * raw_returns[b, :, hi][m].float()).sum())
            if len(rets) >= 2:
                r = torch.stack(rets); per_h.append(-(r.mean() / (r.std() + 1e-6)))
        return torch.stack(per_h).mean() if per_h else scores.new_zeros(())


@torch.no_grad()
def encoder_diagnostics(embeddings: Tensor, valid: Tensor) -> dict:
    """Covariance spectrum globally/within date; effective_rank aliases PR rank.

    Both participation ratio and entropy rank are reported to avoid conflating
    definitions when comparing the earlier audit. The sample rank ceiling also
    matters: a 32-stock date can have at most 31 centered dimensions.
    """
    def metrics(x):
        x = x.float()
        if len(x) < 2:
            return {"effective_rank": float("nan"), "participation_rank": float("nan"),
                    "entropy_rank": float("nan"), "mean_std": float("nan"),
                    "top_eigenvalue_share": float("nan"), "rank_ceiling": max(0, len(x)-1)}
        centered = x - x.mean(0)
        eigen = torch.linalg.svdvals(centered).square() / (len(x) - 1)
        total = eigen.sum()
        if total <= 1e-20:
            return {"effective_rank": 0.0, "participation_rank": 0.0, "entropy_rank": 0.0,
                    "mean_std": 0.0, "top_eigenvalue_share": 1.0, "rank_ceiling": min(len(x)-1, x.shape[1])}
        p = eigen / total
        entropy_rank = torch.exp(-(p * p.clamp_min(1e-30).log()).sum())
        participation_rank = 1 / p.square().sum()
        return {"effective_rank": participation_rank.item(), "participation_rank": participation_rank.item(),
                "entropy_rank": entropy_rank.item(), "mean_std": x.std(0).mean().item(),
                "top_eigenvalue_share": p[0].item(), "rank_ceiling": min(len(x)-1, x.shape[1])}
    global_stats = metrics(embeddings[valid])
    per_date = [metrics(row[mask]) for row, mask in zip(embeddings, valid) if mask.sum() >= 2]
    within = {key: sum(row[key] for row in per_date)/len(per_date) for key in global_stats} if per_date else {}
    return {"global": global_stats, "within_date": within,
            "valid_embeddings": int(valid.sum()), "within_date_count": len(per_date)}
