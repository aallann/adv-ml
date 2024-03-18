import os
import torch
import pandas as pd

from fcnn.model import FCNN
from fcnn.training import Trainer
from fcnn.datasets import load_vanilla_derivative_contracts
from fcnn.training import CallbackLogger, CallbackSaver, CallbackPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, val_loader, data_processor = (
    load_vanilla_derivative_contracts(
        data_path="data",
        contract="call",
        n_train=390000,
        n_test=95000,
        train_batch_size=512,
        test_batch_size=2048,
        n_features=7,
        input_encoder=True,
        output_encoder=False,
    )
)

data_processor.to(device)

model = FCNN(in_dim=7, out_dim=1, hidden_dim=64, n_layers=3, device=device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

Lp1 = torch.nn.L1Loss()
Lp2 = torch.nn.MSELoss()

train_loss = Lp2
eval_loss = {"Lp1": Lp1, "Lp2": Lp2}

trainer = Trainer(
    model=model,
    n_epochs=15,
    data_processor=data_processor,
    callbacks=CallbackLogger(),
    log_freq=1,
    device=device,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loss=train_loss,
    eval_loss=eval_loss,
)

print("Training complete")


# Save the model for deployment

save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "models", "model.pt"
)
torch.save(model.state_dict(), save_path)

# Quicktests


test_samples = test_loader.dataset

for idx in range(3):
    data = test_samples[idx]
    data = data_processor.preprocess(data)

    print(data["x"].shape, data["y"].shape)

    x = data["x"].to(device)
    y = data["y"].to(device)

    output = model(x)

    print(f"Predicted: {output.item()}, True: {y.item()}")


# Validation on implied volatility surface data

eval_samples = val_loader.dataset

predictions = []
for idx in range(len(eval_samples)):
    data = eval_samples[idx]
    data = data_processor.preprocess(data)

    x = data["x"].to(device)
    y = data["y"].to(device)

    prediction = model(x)

    predictions += [prediction.item()]

data_frame = pd.DataFrame(predictions, columns=["IV_NN"])
data_frame.to_csv(f"predictions_singleproccessing.csv")
