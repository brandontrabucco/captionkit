'''Author: Brandon Trabucco, Copyright 2018
Utilities for manipulating images and captions.'''


import os
import os.path
import json
import time
import numpy as np
import tensorflow as tf
import pickle as pkl
from PIL import Image
import string


PUNCTUATION = string.punctuation
UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase
DIGITS = string.digits


def process_string(input_string):
    stage_one = ""
    for character in input_string:
        if character in PUNCTUATION:
            stage_one += " " + character + " "
        if character in UPPER:
            if len(stage_one) > 0 and stage_one[-1] in DIGITS:
                stage_one += " "
            stage_one += character.lower()
        if character == " ":
            stage_one += character
        if character in LOWER:
            if len(stage_one) > 0 and stage_one[-1] in DIGITS:
                stage_one += " "
            stage_one += character
        if character in DIGITS:
            if len(stage_one) > 0 and stage_one[-1] in LOWER:
                stage_one += " "
            stage_one += character
    stage_two = stage_one.replace("  ", " ").replace("  ", " ").strip()
    return stage_two.split(" ")
    
    
def get_unique_words(sentence_generator):
    start_time = time.time()
    unique_words = {}
    for line in sentence_generator:
        tokenized_sentence = process_string(line.strip().split("\t")[0])
        for word in tokenized_sentence:
            if word not in unique_words:
                unique_words[word] = 0
            unique_words[word] += 1
    end_time = time.time()
    print("Finished loading vocabulary, took {0} seconds.".format(end_time - start_time))
    return unique_words


def create_vocab_file(sentence_generator, vocab_filename, min_instances):
    word_dict = get_unique_words(sentence_generator)
    word_list = list(zip(*list(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))))[0]
    for i, word in enumerate(word_list):
        if word_dict[word] < min_instances:
            word_list = word_list[:(i + 1)]
            break
    print("Created a vocabulary with {0} words.".format(len(word_list)))
    with open(vocab_filename, "wb") as f:
        pkl.dump(word_list, f)


def load_image_from_path(image_path):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.float32)


def tile_with_new_axis(tensor, repeats, locations):
    nd = zip(repeats, locations)
    nd = sorted(nd, key=lambda ab: ab[1])
    repeats, locations = zip(*nd)
    for i in sorted(locations):
        tensor = tf.expand_dims(tensor, i)
    reverse_d = {val: idx for idx, val in enumerate(locations)}
    tiles = [repeats[reverse_d[i]] if i in locations else 1 for i, _s in enumerate(tensor.shape)]
    return tf.tile(tensor, tiles)


def collapse_dims(tensor, flat_points):
    flat_size = tf.shape(tensor)[flat_points[0]]
    for i in flat_points[1:]:
        flat_size = flat_size * tf.shape(tensor)[i]
    fixed_points = [i for i in range(len(tensor.shape)) if i not in flat_points]
    fixed_shape = [tf.shape(tensor)[i] for i in fixed_points]
    tensor = tf.transpose(tensor, fixed_points + flat_points)
    final_points = list(range(len(fixed_shape)))
    final_points.insert(flat_points[0], len(fixed_shape))
    return tf.transpose(tf.reshape(tensor, fixed_shape + [flat_size]), final_points)


def remap_decoder_name_scope(var_list):
    var_names = {}
    for x in var_list:
        if "decoder" in x.name and "logits" in x.name:
            var_names[x.name.replace("decoder/", "")
                .replace("decoder_1/", "").replace("decoder_2/", "")
                .replace("decoder_3/", "").replace("decoder_4/", "")[:-2]] = x
        else:
            var_names[x.name.replace("decoder/", "rnn/")
                .replace("decoder_1/", "rnn/").replace("decoder_2/", "rnn/")
                .replace("decoder_3/", "rnn/").replace("decoder_4/", "rnn/")[:-2]] = x
    return var_names