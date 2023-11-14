class CustomRoberta(torch.nn.Module):
    """
    model subclass to define the RoBERTa architecture, also closely based on
    the huggingface tutorial implementation
    """
    def __init__(self, drop_percent, num_classes, pt_model_name: str = 'roberta-base'):
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(pt_model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_percent)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # get outputs from base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # extract hidden state from roberta base outputs
        hidden_state = base_outputs[0]
        x = hidden_state[:, 0]

        # define the linear layer preceding the classifier
        # and apply ReLU activation to its outputs
        x = self.pre_classifier(x)
        x = torch.nn.ReLU()(x)

        # define the dropout layer and classifier
        # and apply Sigmoid activation to its outputs
        x = self.dropout(x)
        x = self.classifier(x)
        outputs = torch.nn.Sigmoid()(x)
        return outputs