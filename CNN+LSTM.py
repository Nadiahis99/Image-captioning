import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
DATASET_PATH = 'Flicker8k_Dataset'
CAPTIONS_FILE = 'Flickr8k.token.txt'
VOCAB_SIZE = 5000
MAX_LENGTH = 34
EPOCHS = 10
BATCH_SIZE = 64

# Step 1: Load image features using VGG16
def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for name in tqdm(os.listdir(directory), desc="Extracting features"):
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[name] = feature[0]
    return features

# Step 2: Load captions with multiple per image
def load_captions(file_path):
    captions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_name, caption = parts
                img_name = img_name.split('#')[0]  # Remove #0, #1, etc.
                if img_name not in captions:
                    captions[img_name] = []
                captions[img_name].append(caption)
    return captions

# Step 3: Clean and prepare text
def process_captions(captions_dict):
    all_captions = []
    for img, caps in captions_dict.items():
        for i in range(len(caps)):
            caption = caps[i].lower().strip()
            caption = f"startseq {caption} endseq"
            captions_dict[img][i] = caption
            all_captions.append(caption)
    return all_captions

# Step 4: Create sequences
def create_sequences(tokenizer, max_length, captions, features, vocab_size):
    X1, X2, y = [], [], []
    for img_name, caption_list in tqdm(captions.items(), desc="Creating sequences"):
        if img_name not in features:
            continue
        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical(out_seq, num_classes=vocab_size)
                feature_vector = features[img_name]
                X1.append(feature_vector)
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Step 5: Define the model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Main execution
if __name__ == '__main__':
    print("Extracting image features...")
    features = extract_features(DATASET_PATH)

    print("Processing text...")
    captions = load_captions(CAPTIONS_FILE)
    all_captions = process_captions(captions)

    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)

    print("Creating training sequences...")
    X1, X2, y = create_sequences(tokenizer, MAX_LENGTH, captions, features, VOCAB_SIZE)
    print(f"Training data shapes: {X1.shape}, {X2.shape}, {y.shape}")

    print("Building and training model...")
    model = define_model(VOCAB_SIZE, MAX_LENGTH)
    model.fit([X1, X2], y, epochs=EPOCHS, batch_size=BATCH_SIZE)
