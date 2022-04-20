import torch
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.span_extractors import EndpointSpanExtractor
import yake

from tne.modeling.span_extractors.yake_span_extractor import YakeSpanExtractor


@SpanExtractor.register("combined_span_extractor")
class CombinedSpanExtractor(SpanExtractor):
    def __init__(self,
                 yake_span_extractor: YakeSpanExtractor,
                 endpoint_span_extractor:EndpointSpanExtractor):
        super().__init__()
        self._yake_span_extractor = yake_span_extractor
        self._endpoint_span_extractor = endpoint_span_extractor

    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            metadata,
            sequence_mask: torch.BoolTensor = None,
            span_indices_mask: torch.BoolTensor = None
    ):
        """
                Given a sequence tensor, extract spans and return representations of
                them. Span representation can be computed in many different ways,
                such as concatenation of the start and end spans, attention over the
                vectors contained inside the span, etc.
                # Parameters
                sequence_tensor : `torch.FloatTensor`, required.
                    A tensor of shape (batch_size, sequence_length, embedding_size)
                    representing an embedded sequence of words.
                span_indices : `torch.LongTensor`, required.
                    A tensor of shape `(batch_size, num_spans, 2)`, where the last
                    dimension represents the inclusive start and end indices of the
                    span to be extracted from the `sequence_tensor`.
                sequence_mask : `torch.BoolTensor`, optional (default = `None`).
                    A tensor of shape (batch_size, sequence_length) representing padded
                    elements of the sequence.
                span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
                    A tensor of shape (batch_size, num_spans) representing the valid
                    spans in the `indices` tensor. This mask is optional because
                    sometimes it's easier to worry about masking after calling this
                    function, rather than passing a mask directly.
                # Returns
                A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
                where `embedded_span_size` depends on the way spans are represented.
        """

        endpoint_span_embeddings = self._endpoint_span_extractor(sequence_tensor, span_indices)
        yake_span_embeddings = self._yake_span_extractor(sequence_tensor, span_indices, metadata)
        yake_span_embeddings = torch.Tensor(yake_span_embeddings).cuda()
        span_embeddings = torch.cat((endpoint_span_embeddings, yake_span_embeddings), dim=2)

        return span_embeddings
