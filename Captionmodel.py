class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, max_len):
        super(ImageCaptioningModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.linear = nn.Linear(self.vit.config.hidden_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.vit(pixel_values=images).last_hidden_state[:, 0, :]
        features = self.linear(features).unsqueeze(1)
        embeddings = self.embedding(captions)
        inputs = torch.cat((features, embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(hiddens)
        return outputs
