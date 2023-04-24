from defense_model import DefenseModelI, DefenseModelII
from data import MNISTData

mnist_data = MNISTData(val_ratio=5/60)

AE_I = DefenseModelI(input_shape=(28, 28, 1), optimizer='adam', noise_volume=0.1, model_path='./defense_models/')
AE_I.train(train_data=mnist_data.x_train,
           val_data=mnist_data.x_val,
           save_name='mnist_defense_model_I',
           epochs=100)

AE_II = DefenseModelII(input_shape=(28, 28, 1), optimizer='adam',
                       noise_volume=0.1, model_path='./defense_models/')
AE_II.train(train_data=mnist_data.x_train,
            val_data=mnist_data.x_val,
            save_name='mnist_defense_model_II',
            epochs=100)


# from data import Cifar10Data
# from defense_model import DefenseModelII
# from custom_data_generator import GaussianDataGenerator
#
# cifar10_data = Cifar10Data(0.1)
#
# train_generator = GaussianDataGenerator(cifar10_data.x_train, batch_size=256)
# val_generator = GaussianDataGenerator(cifar10_data.x_val, batch_size=256)
#
# cifar10_model = DefenseModelII(
#     input_shape=(32, 32, 3),
#     noise_volume=0.025,
#     model_path='defense_models/',
#     optimizer='adam',
#     loss_function='mse'
# )
#
# cifar10_model.defense_model.summary()
#
# cifar10_model.train(
#     train_data=cifar10_data.x_train,
#     val_data=cifar10_data.x_val,
#     save_name='cifar10_defense_model_II_0_5_0_025.h5',
#     epochs=400, batch_size=512
# )

