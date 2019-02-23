#Convolutional neural network

install.packages("keras")
library("Keras")

# Keras -------------------------------------------------------------------

# Maybe it take a while

#install_keras()

mnist <- dataset_mnist()

image(as.matrix(mnist$train$x[2,,]))
mnist$train$y[2]

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

#Reshape removes flatten in FCNN
  
x_train <- x_train %>% array_reshape(c(60000, 28, 28, 1))
x_test <- x_test %>% array_reshape(c(10000, 28, 28, 1))

image(x_train[2,,,])

#Normalize the input (de 0 a 255) a (0 - 1)

x_train <- x_train/255
x_test <- x_test/255


#One-hot encoding a dummy variable

y_train <- to_categorical(y_train, 10)
y_test_real <- y_test
y_test <- to_categorical(y_test, 10)


#Model

modelo_cnn <- keras_model_sequential()

# Lets create the layers!

modelo_cnn %>%
  layer_conv_2d (kernel_size = c(3,3), activation = "relu", filters = 32 , input_shape = c(28, 28, 1),
  layer_conv_2d(kernel_size = c(3,3), activation = "relu", filters = 64 ) %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")


summary(modelo_cnn)


# Cost function
modelo_cnn %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metric = c("accuracy")
  )

resultado <- fit (modelo_cnn, x_train, y_train, epochs = 5,
                  batch_size = 128, validation_split = 0.2)

# "test" evaluation
modelo %>% evaluate(x_test, y_test)


modelo_cnn %>% save_model_hdf5("modelo_cnn_base.hdf5")
modelo_cnn <- load_model_hdf5("modelo_cnn_base.hdf5")


