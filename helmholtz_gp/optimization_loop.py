import torch

def basic_optimization_loop(loss, params, num_iter=10000, learning_rate=0.01):
    best_loss = 1e10
    optimizer = torch.optim.Adam(params.get_unconstrained_params(), lr=learning_rate)
    for step in range(num_iter):
        if step % 100 == 0:
            print(f"At step {step}, loss: {loss(params).item()}")
        optimizer.zero_grad()
        _loss = loss(params)
        _loss.backward()
        if torch.abs(best_loss - _loss) < 1e-4:
            break
        best_loss = _loss
        optimizer.step()

