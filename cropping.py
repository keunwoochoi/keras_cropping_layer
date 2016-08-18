# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec
# imports for backwards namespace compatibility
class Cropping1D(Layer):
    '''Cropping layer for 1D input (e.g. temporal sequence).

    # Arguments
        cropping: tuple of int (length 2)
            How many should be trimmed off at the beginning and end of
            the cropping dimension (axis 1).

    # Input shape
        3D tensor with shape (samples, axis_to_crop, features)

    # Output shape
       3D tensor with shape (samples, cropped_axis, features)

    
    '''

    def __init__(self, cropping=(1,1), **kwargs):
        super(Cropping2D, self).__init__(**kwargs)
        self.cropping = cropping
        self.input_spec = [InputSpec(ndim=3)] # redundant due to build()?       

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def get_output_shape_for(self, input_shape):
        length = input_shape[1] - self.cropping[0][0] - self.cropping[0][1] if input_shape[1] is not None else None
        return (input_shape[0],
                length,
                input_shape[2])

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        return x[:, self.cropping[0][0]:input_shape[1]-self.cropping[0][1], :]

    def get_config(self):
        config = {'cropping': self.padding}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Cropping2D(Layer):
    '''Cropping layer for 2D input (e.g. picture).

    # Arguments
        padding: tuple of tuple of int (length 2)
            How many should be trimmed off at the beginning and end of
            the 2 padding dimensions (axis 3 and 4).
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_crop, second_axis_to_crop)

    # Output shape
        4D tensor with shape:
        (samples, depth, first_cropped_axis, second_cropped_axis)

    
    '''

    def __init__(self, cropping=((0,0),(0,0)), dim_ordering='default', **kwargs):
        super(Cropping2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.cropping = tuple(cropping)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]        

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        return x[:, :, self.cropping[0][0]:input_shape[2]-self.cropping[0][1], self.cropping[1][0]:input_shape[3]-self.cropping[1][1]]

    def get_config(self):
        config = {'cropping': self.padding}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Cropping3D(Layer):
    '''Cropping layer for 2D input (e.g. picture).

    # Arguments
        padding: tuple of tuple of int (length 2)
            How many should be trimmed off at the beginning and end of
            the 2 padding dimensions (axis 3 and 4).
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # Input shape
        5D tensor with shape:
        (samples, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)

    # Output shape
        5D tensor with shape:
        (samples, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)
    
    '''

    def __init__(self, cropping=((1,1),(1,1),(1,1)), dim_ordering='default', **kwargs):
        super(Cropping2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.cropping = tuple(cropping)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]        

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][1] if input_shape[2] is not None else None
            dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][1] if input_shape[3] is not None else None
            dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][1] if input_shape[4] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    dim1,
                    dim2,
                    dim3)
        elif self.dim_ordering == 'tf':
            dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][1] if input_shape[1] is not None else None
            dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][1] if input_shape[2] is not None else None
            dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][1] if input_shape[3] is not None else None
            return (input_shape[0],
                    dim1,
                    dim2,
                    dim3,
                    input_shape[4])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if self.dim_ordering == 'th':
            return x[:, :, self.cropping[0][0]:input_shape[2]-self.cropping[0][1], self.cropping[1][0]:input_shape[3]-self.cropping[1][1], self.cropping[2][0]:input_shape[4]-self.cropping[2][1]]
        elif self.dim_ordering == 'tf':
            return x[:, self.cropping[0][0]:input_shape[1]-self.cropping[0][1], self.cropping[1][0]:input_shape[2]-self.cropping[1][1], self.cropping[2][0]:input_shape[3]-self.cropping[2][1], :]

    def get_config(self):
        config = {'cropping': self.padding}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
