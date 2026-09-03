import torch

from zipvoice.models.zipvoice import ZipVoice

_TINY_KWARGS = dict(
    feat_dim=100,
    fm_decoder_dim=32,
    fm_decoder_num_layers=[1, 1, 1, 1, 1],
    fm_decoder_downsampling_factor=[1, 2, 4, 2, 1],
    fm_decoder_cnn_module_kernel=[31, 15, 7, 15, 31],
    text_encoder_num_layers=1,
    text_encoder_dim=32,
)


def test_scratch_embedding_uses_requested_hidden_size():
    model = ZipVoice(vocab_size=1000, text_embed_dim=64, **_TINY_KWARGS)
    assert model.embed.weight.shape == (1000, 64)
    assert model.text_embed_dim == 64
    # in_dim of text_encoder's own input projection must match the embedding.
    assert model.text_encoder.in_proj.in_features == 64

    embed, lens = model.forward_text_embed([[1, 2, 3, 4], [5, 6]])
    # pad_labels appends one extra pad_id per sequence before padding to the
    # batch max, so the padded seq_len is max(len)+1 here, not max(len).
    assert embed.shape == (2, 5, 100)  # (batch, padded_seq_len, feat_dim)
    assert torch.equal(lens, torch.tensor([4, 2]))


def test_pretrained_embedding_copies_source_weights_no_extra_projection():
    # vocab_size > Qwen2.5-0.5B's base vocab (151936), simulating the
    # [LANG:xx] tokens LanguageModelTokenizer adds on top. text_embed_dim is
    # deliberately set to something else (192) to confirm it's ignored in
    # favor of the pretrained model's actual hidden size (896).
    vocab_size = 151936 + 4
    model = ZipVoice(
        vocab_size=vocab_size,
        text_embed_dim=192,
        embed_source="pretrained",
        pretrained_embed_model="Qwen/Qwen2.5-0.5B",
        **_TINY_KWARGS,
    )
    assert model.embed.weight.shape == (vocab_size, 896)
    assert model.text_embed_dim == 896  # not the requested 192
    assert not hasattr(model, "embed_proj")
    # No separate projection: text_encoder's own in_proj does the 896->32
    # (text_encoder_dim) projection directly.
    assert model.text_encoder.in_proj.in_features == 896
    assert model.text_encoder.in_proj.out_features == 32  # text_encoder_dim

    from transformers import AutoModel

    source_weight = (
        AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")
        .get_input_embeddings()
        .weight.detach()
        .float()
    )
    assert torch.allclose(model.embed.weight[:151936].float(), source_weight)
    # The extra rows for [LANG:xx] tokens are randomly initialized, not copied.
    assert not torch.allclose(model.embed.weight[151936], model.embed.weight[151937])

    embed, lens = model.forward_text_embed([[1, 2, 3, 4], [5, 6]])
    assert embed.shape == (2, 5, 100)  # see the pad_labels note above
    assert torch.equal(lens, torch.tensor([4, 2]))


def test_pretrained_embedding_vocab_smaller_than_source_slices_pretrained_rows():
    # The realistic LanguageModelTokenizer case, found via an actual training
    # run: Qwen2.5-0.5B's real tokenizer vocab is 151,665, but its checkpoint's
    # embedding table has 151,936 rows (padded for hardware alignment). The 4
    # [LANG:xx] tokens land at 151665-151668, still inside that padding, so
    # LanguageModelTokenizer's vocab_size (151669) ends up *smaller* than the
    # source model's embedding table -- every row needed is still a real
    # pretrained one, just a slice of the source table, not extended with any
    # randomly-initialized rows.
    vocab_size = 151669
    model = ZipVoice(
        vocab_size=vocab_size,
        embed_source="pretrained",
        pretrained_embed_model="Qwen/Qwen2.5-0.5B",
        **_TINY_KWARGS,
    )
    assert model.embed.weight.shape == (vocab_size, 896)

    from transformers import AutoModel

    source_weight = (
        AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")
        .get_input_embeddings()
        .weight.detach()
        .float()
    )
    assert torch.allclose(model.embed.weight.float(), source_weight[:vocab_size])

    embed, lens = model.forward_text_embed([[1, 2, 3, 4], [5, 6]])
    assert embed.shape == (2, 5, 100)
    assert torch.equal(lens, torch.tensor([4, 2]))
