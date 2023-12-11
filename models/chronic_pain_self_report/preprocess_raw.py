"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
Modified by Beatrice Brown-Mulry for BMI 550

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import regex as re

def hashtag(text):
    flags = re.MULTILINE | re.DOTALL
    
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=flags))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):
    flags = re.MULTILINE | re.DOTALL
    
    text=str(text)
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl, flags):
        return re.sub(pattern, repl, text, flags=flags)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>", flags)
    text = re_sub(r"@\w+", "<user>", flags)
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>", flags)
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>", flags)
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>", flags)
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", flags)
    text = re_sub(r"/", " / ", flags)
    text = re_sub(r"<3", "<heart>", flags)
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", flags)
    text = re_sub(r"#\S+", hashtag, flags)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>", flags)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", flags)

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps, flags)

    return text.lower()

    
def process_dataframe(df, text_col: str = 'text'):
    # make a copy of the dataframe
    out_df = df.copy()
    
    print('preprocessing dataframe')
    
    # map the preprocessing func to the dataframe
    out_df.loc[:, text_col] = out_df.loc[:, text_col].apply(tokenize)
    
    return out_df
    