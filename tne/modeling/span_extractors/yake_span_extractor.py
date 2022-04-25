import torch
from allennlp.modules.span_extractors import SpanExtractor
import yake


@SpanExtractor.register("yake_span_extractor")
class YakeSpanExtractor(SpanExtractor):
    def __init__(self, method: str, context: str):
        super().__init__()
        self.method = method
        self.context = context

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

        metadata = metadata[0]
        text = metadata['original_text']

        tokens = metadata['tokens']
        context = self.context
        batch_size = sequence_tensor.shape[0]
        doc_length = sequence_tensor.shape[1]
        embedding_size = sequence_tensor.shape[2]
        num_spans = span_indices.shape[1]

        span_embeddings = torch.empty((batch_size, num_spans, embedding_size))

        if context == "doc":
            yake_kw_extractor = yake.KeywordExtractor(n=1, top=doc_length, features=None, stopwords=set())

            # returns list of tuples (word(str), score(float)) by ascending score order
            scores = yake_kw_extractor.extract_keywords(text)

            for i in range(batch_size):
                spans = span_indices[i]
                for j in range(num_spans):
                    span = spans[j]
                    first = span[0]
                    last = span[1]
                    np_tokens = tokens[first:last + 1]
                    word_embeds = sequence_tensor[i][first:last + 1]
                    if len(word_embeds) <= 0:
                        print("\n\n\n\n\n")
                        print("tokens:")
                        print(tokens)
                        print(len(tokens))
                        print(type(tokens))
                        print("\n\n\n\n\n")
                        print("word_embeds:")
                        print(word_embeds)
                        print(len(word_embeds))
                        print(type(word_embeds))
                    np_embedding = self.get_np_embedding(scores, np_tokens, word_embeds)
                    # print("\n\n\n\n\n")
                    # print("np_embedding:")
                    # print(np_embedding)
                    # print(len(np_embedding))
                    # print(type(np_embedding))
                    # span_embeddings[i][j] = np_embedding


        elif context == "np":
            for i in range(batch_size):
                spans = span_indices[i]
                for j in range(num_spans):
                    span = spans[j]
                    first = span[0]
                    last = span[1]
                    np_tokens = tokens[first:last + 1]
                    length = len(np_tokens)
                    noun_phrase = " ".join(np_tokens)
                    yake_kw_extractor = yake.KeywordExtractor(n=1, top=length, features=None, stopwords=set())
                    scores = yake_kw_extractor.extract_keywords(noun_phrase)
                    word_embeds = sequence_tensor[i][first:last + 1]
                    np_embedding = self.get_np_embedding(scores, np_tokens, word_embeds)
                    span_embeddings[i][j] = np_embedding
        else:
            raise Exception("invalid context of yake representation")

        return span_embeddings

    def get_np_embedding(self, scores, np_tokens, word_embeds):
        scores_dict = dict(scores)
        method = self.method

        if method == "wa":
            # with softmax:
            weights = torch.Tensor([-scores_dict.get(word, 0.0) for word in np_tokens])
            softmax = torch.nn.Softmax()
            weights = softmax(weights)
            # without softmax:
            # weights = np.array([1 - scores[word] for word in np_tokens])
            # ones = np.ones_like(weights)
            # zeros = np.zeros_like(weights)
            # weights = np.maximum(zeros, weights)
            # weights = np.minimum(ones, weights)

            # I assumed that word_embeds is a matrix where each row is one word embedding
            # so the average is over the rows
            np_embedding = torch.zeros_like(word_embeds[0])
            for i in range(len(word_embeds)):
                np_embedding += weights[i] * word_embeds[i]
            return np_embedding

        elif method == "wa_top_half":
            length = len(np_tokens)
            np_scores = [(word, -scores_dict.get(word, 0.0)) for word in np_tokens]
            top_half_scores = sorted(np_scores, key=lambda x: x[1])[:length // 2 + 1]
            top_half_words = [tup[0] for tup in top_half_scores]
            top_half_words_indices = [np_tokens.index(word) for word in top_half_words]
            top_half_word_embeddings = [word_embeds[i] for i in top_half_words_indices]
            top_half_scores_dict = dict(top_half_scores)
            weights = torch.Tensor([top_half_scores_dict.get(word, 0.0) for word in top_half_words])

            softmax = torch.nn.Softmax()
            weights = softmax(weights)
            # without softmax:
            # weights = np.array([1 - scores[word] for word in np_tokens])
            # ones = np.ones_like(weights)
            # zeros = np.zeros_like(weights)
            # weights = np.maximum(zeros, weights)
            # weights = np.minimum(ones, weights)

            # I assumed that word_embeds is a matrix where each row is one word embedding
            # so the average is over the rows
            np_embedding = torch.zeros_like(word_embeds[0])
            for i in range(len(top_half_word_embeddings)):
                np_embedding += weights[i] * top_half_word_embeddings[i]
            return np_embedding

        # elif method == "average":
        #     np_embedding = torch.mean(word_embeds, 0)
        #     return np_embedding

        elif method == "max":
            np_scores = [(word, -scores_dict.get(word, 0.0)) for word in np_tokens]
            top = sorted(np_scores, key=lambda x: x[1])[0]
            index = np_tokens.index(top[0])
            np_embedding = word_embeds[index]
            return np_embedding

        # elif method == "wa_first_last":
        #     weights = torch.Tensor([scores_dict.get(np_tokens[0], 0.0), scores_dict.get(np_tokens[-1], 0.0)])
        #     softmax = torch.nn.Softmax()
        #     weights = softmax(weights)
        #     # first_last_embeds = np.array([word_embeds[0], word_embeds[-1]])
        #     # np_embedding = np.average(first_last_embeds, axis=0, weights=weights)
        #     np_embedding = word_embeds[0] * weights[0] + word_embeds[-1] * weights[-1]
        #     return np_embedding

        # elif method == "average_first_last":
        #     # first_last_embeds = np.array([word_embeds[0], word_embeds[-1]])
        #     # np_embedding = np.average(first_last_embeds, axis=0)
        #     np_embedding = word_embeds[0] * 0.5 + word_embeds[-1] * 0.5
        #     return np_embedding

        else:
            raise Exception("invalid method of np embedding creation")
