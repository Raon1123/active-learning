import tqdm

def run_round(train_loader, test_loader, model, optimizer, config):
    r"""Run a single round of training and testing.
    """
    epochs = config['MODEL']['epoch_per_round']
    
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = run_epoch(train_loader, model, optimizer, config['DEVICE'])
    test_loss, test_acc = run_test_epoch(test_loader, model)
        
    return train_loss, test_loss, test_acc


def run_epoch(train_loader, model, optimizer, device='cuda'):
    cum_loss = 0.0
    cnt = 0
    for batch in train_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        
        output = model(data)
        loss = model.loss(output, target)
        loss.backward()
        
        optimizer.step()
        
        cum_loss += loss.item()
        cnt += 1
        
    return cum_loss / cnt


def run_test_epoch(test_loader, model, device='cuda'):
    cum_loss = 0.0
    cum_correct = 0
    cnt = 0
    for batch in test_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        output = model(data)
        loss = model.loss(output, target)
        
        cum_loss += loss.item()
        cum_correct += (output.argmax(dim=1) == target).sum().item()
        cnt += len(data)
    
    return cum_loss / cnt, cum_correct / cnt