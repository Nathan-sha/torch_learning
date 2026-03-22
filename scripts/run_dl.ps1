param(
    [string]$Device = "auto",
    [int]$Epochs = 20,
    [int]$BatchSize = 64
)

python -m src.dl.train_classifier `
    --device $Device `
    --epochs $Epochs `
    --batch-size $BatchSize
