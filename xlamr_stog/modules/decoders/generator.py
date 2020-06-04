import torch

from xlamr_stog.metrics.seq2seq_metrics import Seq2SeqMetrics

class Generator(torch.nn.Module):

    def __init__(self, input_size, vocab_size, vocab_pad_idx):
        super(Generator, self).__init__()
        self._generator = torch.nn.Sequential(
            torch.nn.Linear(input_size, vocab_size),
            torch.nn.LogSoftmax(dim=-1)
        )
        self.criterion = torch.nn.NLLLoss(
            ignore_index=vocab_pad_idx, reduction='sum'
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.metrics = Seq2SeqMetrics()
        self.vocab_pad_idx = vocab_pad_idx
        self.input_size = input_size
        self.vocab_size = vocab_size


    def forward(self, inputs):
        """Transform inputs to vocab-size space and compute logits.

        :param inputs:  [batch, seq_length, input_size]
        :return:  [batch, seq_length, vocab_size]
        """
        batch_size, seq_length, _ = inputs.size()
        inputs = inputs.view(batch_size * seq_length, -1)
        scores = self._generator(inputs)
        scores = scores.view(batch_size, seq_length, -1)
        probs = self.softmax(scores)
        _, predictions = probs.max(2)
        return dict(
            probs=probs,
            scores=scores,
            predictions=predictions,
            source_dynamic_vocab_size = 0,
            target_dynamic_vocab_size= 0,
        )

    def compute_loss(self, inputs, targets):
        batch_size, seq_length, _ = inputs.size()
        output = self(inputs)
        scores = output['scores'].view(batch_size * seq_length, -1)
        predictions = output['predictions'].view(-1)
        targets = targets.view(-1)

        loss = self.criterion(scores, targets)

        non_pad = targets.ne(self.vocab_pad_idx)
        num_correct = predictions.eq(targets).masked_select(non_pad).sum().item()
        num_non_pad = non_pad.sum().item()
        self.metrics(loss.item(), num_non_pad, num_correct)

        return dict(
            loss=loss.div(float(num_non_pad)),
            total_loss=loss.sum(),
            num_tokens=torch.tensor([float(num_non_pad)]).type_as(loss),
            predictions=output['predictions']
        )

    @classmethod
    def from_params(cls, params):
        return cls(
            input_size=params['input_size'],
            vocab_size=params['vocab_size'],
            vocab_pad_idx=params['vocab_pad_idx']
        )
