
def create_model(opt):
    model = None
    
    if opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [{}] not recognized.".format(opt.model))
    
    model.initialize(opt)
    print("model [{}] was created".format(model.name()))

    return model