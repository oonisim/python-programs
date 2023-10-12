"""
Originated from https://github.com/Trindaz/EFZP/blob/master/EFZP.py.

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""
import regex as re
import string


def parse(text, remove_quoted_statements=False):
    text = text.strip()
    text = strip_ocr_page_header(text)
    text = strip_automated_notation(text)
    if remove_quoted_statements:
        pattern = """(?P<quoted_statement>".*?")"""
        matches = re.findall(pattern, text, re.IGNORECASE + re.DOTALL)
        for m in matches:
            text = text.replace(m, '"[quote]"')
    result = {
        "salutation":get_salutation(text),
        "body":get_body(text),
        "signature":get_signature(text),
        "reply_text":get_reply_text(text)
    }
    return result


def strip_ocr_page_header(text: str) -> str:
    """Remove Page n of N from ATT m of M if the text start with it.
    """
    if re.match(
        pattern=r"^p?age \d+ of \d+ from ATT \d+ of \d+",
        string=text,
        flags=re.IGNORECASE
    ):
        text = re.sub(
            pattern=r"p?age \d+ of \d+ from ATT \d+ of \d+\s*",
            repl="\n\n",
            string=text,
            flags=re.IGNORECASE
        )

    return text.lstrip()


# automated_notation could be any labels or sections in the email giving special notation for
# human readers of the email text. For example, text may start with "A message from your customer:"
def strip_automated_notation(text):
    # Use a parameter name text to indicate text in the actual email message
    notations = [
        r"Hi, there has been a new enquiry from\..*?Enquiry:(?P<email_message>.*)",
    ]
    for n in notations:
        groups = re.match(n, text, re.IGNORECASE + re.DOTALL)
        if groups is not None:
            # if groups.groupdict().has_key("email_message"):
            if "email_message" in groups.groupdict():
                text = groups.groupdict()["email_message"]
    
    return text
    

def get_reply_text(text):
    # Notes on regex
    # Search for classic prefix from GMail and other mail clients "On May 16, 2011, Dave wrote:"
    # Search for prefix from outlook clients From: Some Person [some.person@domain.tld]
    # Search for prefix from outlook clients when used for sending to recipients in
    # the same domain From: Some Person\nSent: 16/05/2011 22:42\nTo: Some Other Person
    # Search for prefix when message has been forwarded
    # Search for From: <email@domain.tld>\nTo: <email@domain.tld>\nDate:<email@domain.tld
    # Search for From:, To:, Sent:
    # Some clients use -*Original Message-*
    pattern = r"(?P<reply_text>" + \
        r"On ([a-zA-Z0-9, :/<>@\.\"\[\]]* wrote:.*)|" + \
        r"From: [\w@ \.]* \[mailto:[\w\.]*@[\w\.]*\].*|" + \
        r"From: [\w@ \.]*(\n|\r\n)+Sent: [\*\w@ \.,:/]*(\n|\r\n)+To:.*(\n|\r\n)+.*|" + \
        r"[- ]*Forwarded by [\w@ \.,:/]*.*|" + \
        r"From: [\w@ \.<>\-]*(\n|\r\n)To: [\w@ \.<>\-]*(\n|\r\n)Date: [\w@ \.<>\-:,]*\n.*|" + \
        r"From: [\w@ \.<>\-]*(\n|\r\n)To: [\w@ \.<>\-]*(\n|\r\n)Sent: [\*\w@ \.,:/]*(\n|\r\n).*|" + \
        r"From: [\w@ \.<>\-]*(\n|\r\n)To: [\w@ \.<>\-]*(\n|\r\n)Subject:.*|" + \
        r"(-| )*Original Message(-| )*.*)"
    groups = re.search(pattern, text, re.IGNORECASE + re.DOTALL)
    reply_text = None
    if groups is not None:
        # if groups.groupdict().has_key("reply_text"):
        if "reply_text" in groups.groupdict():
            reply_text = groups.groupdict()["reply_text"]
    return reply_text
    

def get_signature(text):
    
    # try not to have the signature be the very start of the message if we can avoid it
    salutation = get_salutation(text)
    if salutation: text = text[len(salutation):]
    
    # note - these opening statements *must* be in lower case for
    # sig within sig searching to work later in this func
    sig_opening_statements = [
        "please do not hesitate",
        "regards",
        "warm regards",
        "kind regards",
        "best regards",
        "with warm regards",
        "with kind regards",
        "with best regards",
        "all the best"
        "cordially",
        "yours cordially",
        "yours truly",
        "cheers",
        "sincerely",
        "yours sincerely",
        "faithfully",
        "yours faithfully",
        "ciao",
        "bye",
        "talk soon",
        "thank (you)? again for referring",
        "thanks again for referring",
        "sent from my iphone"
    ]
    # pattern = "(?P<signature>(" + string.join(sig_opening_statements, "|") + ")(.)*)"
    pattern = rf"(?P<signature>({'|'.join(sig_opening_statements)})(.)*)"
    groups = re.search(pattern, text, re.IGNORECASE + re.DOTALL)
    signature = None
    if groups:
        if "signature" in groups.groupdict():
            signature = groups.groupdict()["signature"]
            reply_text = get_reply_text(text[text.find(signature):])
            if reply_text: signature = signature.replace(reply_text, "")
            
            # search for a sig within current sig to lessen chance of accidentally stealing words from body
            tmp_sig = signature
            for s in sig_opening_statements:
                if tmp_sig.lower().find(s) == 0:
                    tmp_sig = tmp_sig[len(s):]
            groups = re.search(pattern, tmp_sig, re.IGNORECASE + re.DOTALL)
            if groups:
                signature = groups.groupdict()["signature"]
        
    # if no standard formatting has been provided (e.g. Regards, <name>),
    # try a probabilistic approach by looking for phone numbers, names etc. to derive sig
    if not signature:
        # body_without_sig = get_body(text, check_signature=False)
        pass
    
    # check to see if the entire body of the message has been 'stolen' by the signature.
    # If so, return no sig so body can have it.
    body_without_sig = get_body(text, check_signature=False)
    if signature==body_without_sig: signature = None
    
    return signature


# todo:
#  complete this out (I bit off a bit more than I could chew with this function.
#  Will probably take a bunch of basian stuff
def is_word_likely_in_signature(word, text_before="", text_after=""):
    # Does it look like a phone number?
    
    # is it capitalized?
    if word[:1] in string.ascii_uppercase and word[1:2] in string.ascii_lowercase: return True
    
    return


# check_<zone> args provided so that other functions can call get_body without causing infinite recursion
def get_body(text, check_salutation=True, check_signature=True, check_reply_text=True):
    
    if check_salutation:
        sal = get_salutation(text)
        if sal:
            # text = text[text.find(sal) + len(sal):]
            text = text[text.find(sal):]

    if check_signature:
        sig = get_signature(text)
        if sig:
            text = text[:text.find(sig)]
    
    if check_reply_text:
        reply_text = get_reply_text(text)
        if reply_text:
            text = text[:text.find(reply_text)]
            
    return text


def get_salutation(text):
    """remove reply text fist (e.g. Thanks\nFrom: email@domain.tld causes salutation to consume start of reply_text
    """
    reply_text = get_reply_text(text)
    if reply_text:
        text = text[:text.find(reply_text)]
    # Notes on regex:
    # Max of 5 words succeeding first Hi/To etc, otherwise is probably an entire sentence
    openings = [
        "re:",
        "thank you for referring",
        "thank you for (your|the) referral of",
        "thanks for (your|the) referral of",
        "good morning",
        "good afternoon",
        "hello",
        "hi",
        "dear",
    ]
    salutations = [
        "dr",
        "mr.",
        "mr",
        "mrs.",
        "mrs",
        "miss"
    ]

    pattern = (
        rf"((?<!\S)({'|'.join(openings)})(?!\S)\s*)+" +
        rf"(\s*(?<!\S)({'|'.join(salutations)})(?!\S))?" +
        rf"(\s*\w*)(\s*\w*)(\s*\w*)(\s*\w*)(\s*\w*)([.\s]*)"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    salutation = None
    if match is not None:
        # if "salutation" in groups.groupdict():
        #     salutation = groups.groupdict()["salutation"]
        salutation = match.group(0)

    return salutation
    
