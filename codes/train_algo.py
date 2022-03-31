from helper import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="LeNet5", choices=["LeNet5"], help='set the model name')
parser.add_argument('-al', '--algorithm', default="SWAG", choices=["FL-SWAG", "SWAG", "SGLD", "pSGLD"], help='set the optimization algorithm')
parser.add_argument('-n', '--num_epochs', default=10, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_name', default="FashionMNIST", choices=["MNIST", "FashionMNIST"], help='set the dataset')
parser.add_argument('-l', '--learning_rate', default=1e-2, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=256, type=int, help='set the mini-batch size')
parser.add_argument('-w', '--weight_decay', default=0.0, type=float, help='set the parameter of the gaussian prior')
parser.add_argument('-b', '--t_burn_in', default=0, type=int, help='set the burn in period')
parser.add_argument('-s', '--seed', default=1, type=int, help='set the seed')
parser.add_argument('-pre', '--pretrained', default=False, type=bool, help='indicates whehter the same model has been trained before or not.')
args = parser.parse_args()

title_dictionary = {'d_': args.dataset_name, 'l_': args.learning_rate, 'weight_decay_': args.weight_decay}
title = "-".join([key + str(value) for key, value in title_dictionary.items()])

path = os.path.abspath(".")
path_dataset = path + '/data'
path_figures = path + '/figures'
path_variables = path + '/variables'
path_ckpts = path + '/ckpts'

# Create the directory if it does not exist
os.makedirs(path_dataset, exist_ok=True)
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)
os.makedirs(path_ckpts, exist_ok=True)

seed = args.seed
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the function associated with the chosen dataset
dataset = getattr(torchvision.datasets, args.dataset_name)

batch_size = args.batch_size
train_dataset = dataset(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = dataset(root='./data', train=False, download=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

model_cfg = models.LeNet5MNIST
model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

if not args.pretrained:
    wd = 0.
    lr_init = 1e-1
    train(model, train_loader, test_loader, optimizer, criterion, lr_init, title=title, epochs=20)
else:
    model.load_state_dict(torch.load("ckpts/" + title + ".pt"))


print(f"Training Model {args.model} with {args.algorithm}")
lr_init = args.learning_rate
wd = args.weight_decay
epochs = args.num_epochs
new_title = title + args.algorithm
if args.algorithm == "SWAG":
    swag_model = SWAG(model_cfg.base, subspace_type="pca", *model_cfg.args, **model_cfg.kwargs, 
                  subspace_kwargs={"max_rank": 2, "pca_rank": 2})
    # We do not want to change the weights obtained by normal training.
    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
    model.load_state_dict(torch.load("ckpts/" + title + ".pt"))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, weight_decay=wd)
    
    #test_loader is only included to display accuracy
    train(model, train_loader, test_loader, optimizer, criterion, lr_init, epochs, title=new_title, print_freq=5, 
          swag=True, swag_model=swag_model, swag_start=2, swag_freq=1, swag_lr=1e-2)
    all_probs = model_averaging(swag_model, model=model_cfg.base(*model_cfg.args, **model_cfg.kwargs), loader=test_loader)
    
if args.algorithm == "SGLD" or args.algorithm == "pSGLD":
    path = "ckpts/" + new_title + ".pt"
    algo_wd = wd
    t_burn_in = args.t_burn_in
    sgld = Sgld(model)
    if args.algorithm == "SGLD":
        state_dict, save_dict = sgld.run(train_loader, test_loader, epochs, params_optimizer={'lr' : lr_init}, weight_decay=algo_wd, t_burn_in=t_burn_in, path_save_samples=path)
    elif args.algorithm == "SGLD":
        state_dict, save_dict = sgld.run(train_loader, test_loader, epochs, params_optimizer={'lr' : lr_init, 'precondition_decay_rate' : 0.95}, weight_decay=algo_wd, t_burn_in=t_burn_in, path_save_samples=path)

    all_probs = np.array(sgld_tools.predictions(test_loader, model, path=path, device=device))
    

    
    
save_dict = vars(args)
ytest = np.array(test_loader.dataset.targets)

# Compute the final accuracy of the averaged model from all_probs(bayesian learning)
final_acc = accuracy_all_probs(all_probs, ytest)

# Compute the accuracy in function of p(y|x)>tau
tau_list = np.linspace(0, 1, num=100)
accuracies, misclassified = confidence(ytest, all_probs, tau_list)

# Compute the Expected Calibration Error (ECE)
ece = ECE(all_probs, ytest, num_bins = 20)

# Compute the Brier Score
bs = BS(ytest, all_probs)

# Compute the accuracy - confidence
acc_conf = accuracy_confidence(all_probs, ytest, tau_list, num_bins = 20)

# Compute the calibration curve
cal_curve = calibration_curve(all_probs, ytest, num_bins = 20)

# Save the statistics
save_dict["ytest"] = ytest
save_dict["tau_list"] = tau_list
save_dict["all_probs"] = all_probs
save_dict["accuracies"] = accuracies
save_dict["calibration_curve"] = cal_curve
save_dict["accuracy_confidence"] = acc_conf
# save_dict["save_stats"] = save_stats
torch.save(save_dict, path_variables + '/' + args.algorithm + new_title)


plt.plot(tau_list, acc_conf)
plt.savefig(path_figures + '/acc_conf-' + new_title + '.pdf', bbox_inches='tight')
plt.plot(cal_curve[1], cal_curve[1] - cal_curve[0])
plt.savefig(path_figures + '/cal_curve-' + new_title + '.pdf', bbox_inches='tight')


# Compute the Negative Log Likelihood (NLL)
sample_list = []
path_save_samples = "./ckpts"
for f in os.listdir(path_save_samples):
    net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
    net.load_state_dict(torch.load(os.path.join(path_save_samples, f)))
    net.to(device)
    sample_list.append(copy.deepcopy(net))
entropy_dict, nll = Predictive_entropy(ytest, all_probs, PostNet(sample_list), test_loader, path_dataset)

# Compute the Area Under the Curve (AUC)
auc = AUC(entropy_dict["Dataset"], entropy_dict["OOD dataset"])

# Save the statistics
save_dict["entropy_dict"] = entropy_dict
torch.save(save_dict, path_variables + '/' + args.algorithm + new_title)

# Store the ECE, BS, NNL, AUC
file = open(path_variables + "/text" + new_title + '.txt', 'a')
file.write(f"\nFinal accuracy = {final_acc}, \nECE = {ece}, \nBS = {bs}, \nNLL = {nll}, \nAUC = {auc}")
file.close()  # to change the file access mode








