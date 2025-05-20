device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(tokenizer.word_index) + 3

model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=vocab_size, max_len=max_len).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/satiliteimagecaption/model_vit_beam.pth'))
preds, targets = evaluate_model(model, valid_loader, tokenizer, device=device)
