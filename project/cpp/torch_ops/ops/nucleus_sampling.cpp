#include "nucleus_sampling.h"

namespace torch_ops::ops::nucleus_sampling {

/**
 * Nucleus sampling function to use in TorchScript graph.
 *
 * @param logits The probability distribution tensor, with shape (B, C)
 * @param nucleus_prob The nucleus sampling minimum probability
 * @returns The one-hot next token, with shape (B, C)
 */
torch::Tensor nucleus_sampling(torch::Tensor logits, double nucleus_prob) {
  TORCH_CHECK(logits.dim() == 2, "Expected `logits` to have 2 dimensions, got ",
              logits.dim());
  if (logits.isnan().any().item().toBool()) {
    torch::TensorOptions options = logits.options().dtype(c10::kLong);
    return torch::randint(logits.size(-1), {logits.size(0), 1}, options);
  }
  auto [sorted_logits, inds] = logits.sort(1, /* descending */ true);
  auto sorted_probs = sorted_logits.softmax(1);
  auto probs_cum_sum = sorted_probs.cumsum(1);
  auto min_prob = probs_cum_sum.slice(/* dim */ 1, /* start */ 0, /* end */ 1,
                                      /* step */ 1) +
                  nucleus_prob;
  auto mask_inds = inds.masked_select(probs_cum_sum > min_prob);
  auto masked_logits = logits.index_fill(1, mask_inds, -1e4);
  auto probs = masked_logits.softmax(1);
  return torch::multinomial(probs, 1);
}

void add_torch_module(torch::Library &m) {
  m.def("nucleus_sampling", nucleus_sampling);
}

} // namespace torch_ops::ops::nucleus_sampling
