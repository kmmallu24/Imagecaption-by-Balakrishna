def evaluate_model(model, dataloader, tokenizer, device='cuda'):
    model.eval()
    predictions = []
    actuals = []
    for images, captions in tqdm(dataloader):
        for i in range(images.size(0)):
            image = images[i].to(device)
            generated = beam_search(model, image, tokenizer, device=device)
            predicted_caption = tokenizer.sequences_to_texts([generated])[0]
            actual_caption = tokenizer.sequences_to_texts([captions[i].tolist()])[0]
            predictions.append(predicted_caption)
            actuals.append(actual_caption)
    return predictions, actuals
