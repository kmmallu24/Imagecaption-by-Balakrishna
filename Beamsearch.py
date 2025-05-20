def beam_search(model, image, tokenizer, beam_size=3, max_len=30, device='cuda'):
    with torch.no_grad():
        features = model.vit(pixel_values=image.unsqueeze(0).to(device)).last_hidden_state[:, 0, :]
        features = model.linear(features)

    sequences = [[list(), 0.0]]
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if len(seq) > 0 and seq[-1] == 2:
                all_candidates.append((seq, score))
                continue
            input_seq = torch.tensor([1] + seq).unsqueeze(0).to(device)
            input_embeds = model.embedding(input_seq)
            features_expand = features.unsqueeze(1)
            inputs = torch.cat((features_expand, input_embeds), 1)
            output, _ = model.lstm(inputs)
            preds = model.fc(output[:, -1, :])
            probs = torch.nn.functional.log_softmax(preds, dim=1)
            topk = torch.topk(probs, beam_size, dim=1)
            for i in range(beam_size):
                candidate = seq + [topk.indices[0][i].item()]
                candidate_score = score + topk.values[0][i].item()
                all_candidates.append((candidate, candidate_score))
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_size]
    return sequences[0][0]
