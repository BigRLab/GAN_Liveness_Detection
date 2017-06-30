import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.logger import Logger 

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

logger = Logger(opt)

model = create_model(opt)

total_steps = 0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1) # iter index in current epoch
        
        model.set_input(data)
        model.optimize_parameters()

        # print and log progress information
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            logger.print_current_errors(epoch, epoch_iter, errors, t)

        # save the latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch {}, total_steps {})'.format(epoch, total_steps))
            model.save('latest') # 'latest' is the epoch_label

    # save the model after each epoch
    if total_steps % opt.save_latest_freq == 0:
        print('saving the latest model (epoch {}, total_steps {})'.format(epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    if epoch > opt.niter:
        model.update_learning_rate()