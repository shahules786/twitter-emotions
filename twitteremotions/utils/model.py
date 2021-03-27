import tensorflow as tf
from transformers import TFRobertaModel, RobertaConfig


def emotion_model(path="data/tf_roberta/", maxlen=168):

    ids = tf.keras.layers.Input((maxlen,), dtype=tf.int32)
    att = tf.keras.layers.Input((maxlen,), dtype=tf.int32)
    tok = tf.keras.layers.Input((maxlen,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(path + "config-roberta-base.json")
    bert_model = TFRobertaModel.from_pretrained(path + "pretrained-roberta-base.h5", config=config)
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    # END INDEX HEAD
    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2b = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2b)
    x2 = tf.keras.layers.Activation("softmax")(x2)

    # START INDEX HEAD
    x1 = tf.keras.layers.Concatenate()([x2b, x[0]])
    x1 = tf.keras.layers.Dropout(0.1)(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation("softmax")(x1)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    return model
