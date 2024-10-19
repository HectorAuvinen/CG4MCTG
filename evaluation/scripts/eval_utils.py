import re


# filename parsing
def parse_config_from_filename(filename):
    params = {
        'method': None,
        'rf': None,
        'af': None,
        'dout': None,
        'lr': None,
        'ln': None,
        'ln_res': None,
        'lambda': None,
        'plen': None,
        'secondary_method': None,
        'dataset': None,
        'protocol': None,
        'batch_size': None,
        'epoch': None,
        'seed': None,
        'test_seed': None,
        'ignore_index': None,
        'data_split': None,  # To store 'seen' or 'unseen'
    }

    known_keys = [
        'rf', 'af', 'dout', 'lr', 'ln', 'ln_res', 'lambda', 'plen',
        'bs', 'batch_size', 'epoch', 'seed', 'test_seed', 'ignore_index'
    ]
    known_secondary_methods = [
        'dcg', 'dcg_adapter', 'ctrl', 'contrastive_prefix', 'prefix_tuning'
    ]
    known_datasets = ['Mixture', 'Yelp', 'Amazon', 'Fyelp']
    known_protocols = ['Original', 'Few-Shot', 'Hold-Out', 'Few', 'Hold']

    # Build the regex pattern to match key-value pairs
    all_keys = known_keys + known_secondary_methods + known_datasets + known_protocols
    all_keys_pattern = '|'.join(map(re.escape, all_keys))

    # Pattern to match key-value pairs
    pattern = re.compile(
        r'(?P<key>' + all_keys_pattern + r')'  # Match the key
        r'(?:-|=)'                              # Separator (- or =)
        r'(?P<value>.*?)'                       # Non-greedy match for the value
        r'(?=-' + all_keys_pattern + r'[-=]|$)' # Lookahead for the next key or end
    )

    # Initialize the start position
    pos = 0

    # Extract 'seen' or 'unseen' status at the end
    data_split_match = re.search(r'(?:_|-)(seen|unseen)(?:\.jsonl)?$', filename)
    if data_split_match:
        params['data_split'] = data_split_match.group(1)
        filename = filename[:data_split_match.start()]

    # Extract method (everything before the first key)
    method_match = re.match(r'^(.*?)(?=-' + all_keys_pattern + r'[-=]|$)', filename)
    if method_match:
        params['method'] = method_match.group(1).strip('-')
        pos = len(method_match.group(0))
    else:
        params['method'] = filename.strip('-')
        return params  # No keys found, return early

    # Now, iterate over the matches
    for match in pattern.finditer(filename, pos):
        key = match.group('key')
        value = match.group('value').rstrip('-').strip()

        # Convert value to appropriate type
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            try:
                if re.match(r'^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$', value, re.IGNORECASE):
                    value = float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                pass  # Keep as string

        if key == 'bs' or key == 'batch_size':
            params['batch_size'] = value
        elif key in ['seed', 'test_seed', 'epoch', 'ignore_index']:
            params[key] = value
        elif key in known_keys:
            params[key] = value
        elif key in known_secondary_methods:
            params['secondary_method'] = key
        elif key in known_datasets:
            params['dataset'] = key
        elif key in known_protocols:
            if key == 'Few' and value == 'Shot':
                params['protocol'] = 'Few-Shot'
            elif key == 'Hold' and value == 'Out':
                params['protocol'] = 'Hold-Out'
            else:
                params['protocol'] = key
        else:
            print("Unkown key: ", key)
            pass  # Unknown key

            
    params = {k: v.strip("-") if isinstance(v,str) else v for k, v in params.items()}
    return params


def sanitize_filename(name):
    # Replace any invalid character with an underscore
    return re.sub(r'[^a-zA-Z0-9\-_.]', '_', name)