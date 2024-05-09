import tensorflow as tf
import os
from tensorflow.keras import layers 
from tensorflow.keras.layers import Flatten

def WaveNetResidual(num_filters, kernel_size, dilation_rate):        
    """A WaveNet-like residual block."""
    def build_residual_block(x_init):
        tanh_out = layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='tanh')(x_init)
        sigm_out = layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(x_init)
        x = layers.Multiply()([tanh_out, sigm_out])
        x = layers.Conv1D(num_filters, 1, padding='causal')(x)
        res_x = layers.Add()([x, x_init])
        return res_x, x #residual and skip connections
    return build_residual_block

def build_wavenet_model_residual_blocks(input_shape, num_classes, num_filters, numberOfBlocks, numberOfResidualsPerBlock, kernel_size=2):
    inputs = layers.Input(shape=input_shape)
    # Initial causal convolution layer
    x=layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=1, padding='causal', activation='relu', input_shape=input_shape)(inputs)

    totalLayers=numberOfBlocks*numberOfResidualsPerBlock  
    # Stacked residual blocks
    for i in range(totalLayers):
        k=i%numberOfResidualsPerBlock #e.g. with 3 block and 10 residuals will be 2^0,..2^9,2^0,..2^9,2^0,..2^9
        x, skip = WaveNetResidual(num_filters, kernel_size, 2**k)(x)
        if i==0:
            skips = skip
        else:
            skips = layers.Add()([skips, skip])

    # Classification layers
    x = layers.Activation('relu')(skips)
    x = layers.Conv1D(num_filters,1, activation='relu')(x)
    x = layers.Conv1D(num_filters,1, activation='relu')(x)
    x = layers.Dense(num_filters, activation='relu')(x)    
    x = Flatten()(x)   
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def WavenetRun(model,filters,batch_size,epochs,learning_rate,numberOfResidualsPerBlock,numberOfBlocks,width,classes, X_train, X_val,  X_test, y_train, y_val, y_test):  
    # Define input shape and number of classes
    input_shape = (width, 1)  
    num_classes = len(classes) 
    
    if(model is None):
        model = build_wavenet_model_residual_blocks(input_shape, num_classes, filters, numberOfBlocks, numberOfResidualsPerBlock)    
        
    from tensorflow.keras.optimizers import Adam

    custom_learning_rate = learning_rate

    custom_adam_optimizer = Adam(learning_rate=custom_learning_rate)

    model.compile(optimizer=custom_adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    #Model will be saved at each iteration in a different file in categorical order
    def get_checkpoint_filename(base_filename):
        i = 1
        while True:
            checkpoint_name = f"{base_filename[:-4]}{i}.h5"
            if not os.path.exists(checkpoint_name):                
                return checkpoint_name
            i += 1

    base_filename = f'Wavenet{str(filters)}_{str(numberOfBlocks)}_{str(numberOfResidualsPerBlock)}_{str(width)}_1.h5'
    final_checkpoint_name = get_checkpoint_filename(base_filename)
    base_filename = final_checkpoint_name

    # Callbacks 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=1, mode='min'),
        tf.keras.callbacks.ModelCheckpoint(final_checkpoint_name, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    ]

    # Train the model     
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val) ,callbacks=callbacks,verbose=1)  
        
    # Save the model history
    with open(f'./History/{base_filename.split(".")[0]}.txt',"w") as file:
        file.write(str(history.history))          
    
