from cifar10_utils import *
from torchsummary import summary

def main():
    args = retrieve_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    base_model_path = "examples/cifar10/cifar10_vgg_9510.pt"
    new_base_model_path = "examples/cifar10/new_base.bin"
    snn_output_npy_path = "examples/cifar10/outputs_.npy"
    snn_loop_output_npy_path = "examples/cifar10/outputs_loop.npy"

    # try:
    #     model = torch.load(new_base_model_path)
    #     model.eval()
    # except:
    #     exit()
    
    # evaluate_on_snn(
    #     create_base_from(model, device), 
    #     args.T, 
    #     generate_output_npy=False, 
    #     save_path="", 
    #     device=device)
    
    # exit()

    if(args.from_scratch == 1):
        model = create_base()
        train_on_cifar10(model, args.batch_size, args.k, args.lr, args.gamma, args.epochs, args.log_interval, device)
    else:
        try:
            model = create_base()
            model.load_state_dict(torch.load(base_model_path), strict=False)
            model.eval()
        except:
            print("couldn't load base model")
            exit()

    model.cpu()
    summary(model, (3,32,32), device="cpu")
    model.to(device)

    #warm up
    train_on_cifar10(model, args.batch_size, aug_k=3, lr=args.lr, step_gamma=args.gamma, epochs=1, log_interval=args.log_interval, device=device)
    
    # 0
    evaluate_on_snn(
        create_base_from(model, device), 
        args.T, 
        generate_output_npy=True, 
        save_path=snn_loop_output_npy_path, 
        device=device)

    # 1
    freeze_base(model, device)
    augment_model(model, device)

    # 2
    model.eval()
    train_on_snn_output(model, np.load(snn_loop_output_npy_path), args.batch_size, args.lr, args.gamma, args.epochs, args.log_interval, device)

    # 3
    validate_base_vs_tuned(model, base_model_path, device)

    # 4 
    unfreeze_base(model, device)
    freeze_support(model, device)
    train_on_cifar10(model, args.batch_size, args.k, args.lr, args.gamma, args.epochs, args.log_interval, device)
    save_as_new_base(model, new_base_model_path, device)

    # 5
    evaluate_on_snn(
        create_base_from(model, device), 
        args.T, 
        generate_output_npy=True, 
        save_path=snn_loop_output_npy_path, 
        device=device)

    # LOOP NOW (meditative repression)
    for i in range(args.loops):
        try:
            model = torch.load(new_base_model_path)
            model.eval()
        except:
            exit()

        # 1
        freeze_base(model, device)
        augment_model(model, device)
        unfreeze_support(model, device)

        # 2
        model.eval()
        scaled_epochs = args.epochs
        train_on_snn_output(model, np.load(snn_loop_output_npy_path), args.batch_size, args.lr, args.gamma, scaled_epochs, args.log_interval, device)

        # 3
        validate_base_vs_tuned(model, new_base_model_path, device)

        # 4 
        unfreeze_base(model, device)
        freeze_support(model, device)
        train_on_cifar10(model, args.batch_size, args.k, args.lr, args.gamma, args.epochs, args.log_interval, device)
        save_as_new_base(model, new_base_model_path, device)

        # 5
        evaluate_on_snn(
            create_base_from(model, device), 
            args.T, 
            generate_output_npy=True, 
            save_path=snn_loop_output_npy_path, 
            device=device)

    evaluate_on_snn(
        create_base_from(model, device), 
        args.T, 
        generate_output_npy=False, 
        save_path="", 
        device=device)

if __name__ == '__main__':
    main()
    
