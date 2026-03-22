param(
    [ValidateSet("dqn", "ppo", "sb3-dqn")]
    [string]$Algo = "dqn",
    [string]$EnvId = "CartPole-v1",
    [string]$Device = "auto",
    [int]$TotalSteps = 20000
)

if ($Algo -eq "dqn") {
    python -m src.rl.train_dqn `
        --env $EnvId `
        --device $Device `
        --total-steps $TotalSteps
}
elseif ($Algo -eq "ppo") {
    python -m src.rl.train_sb3 `
        --env $EnvId `
        --algo ppo `
        --device $Device `
        --total-steps $TotalSteps
}
else {
    python -m src.rl.train_sb3 `
        --env $EnvId `
        --algo dqn `
        --device $Device `
        --total-steps $TotalSteps
}
