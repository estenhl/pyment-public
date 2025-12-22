import argparse
import base64
import os
import requests


def upload_weights_to_github(filename: str, token: str, user: str, repo: str):
    with open(filename, 'rb') as f:
        bytes = f.read()

    bytes = base64.b64encode(bytes).decode()

    if os.path.isfile(token):
        with open(token, 'r') as f:
            token = f.read().strip()

    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {token}',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    content = {
        'content': bytes,
        'encoding': 'base64'
    }

    url = f'https://api.github.com/repos/{user}/{repo}/git/blobs'

    response = requests.post(url, json=content, headers=headers)

    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Uploads a weights-file to github')

    parser.add_argument(
        '-f', '--filename',
        required=True,
        help='Path to file containing weights'
    )
    parser.add_argument(
        '-t', '--token',
        required=True,
        help='Token for the GitHub API'
    )
    parser.add_argument(
        '-u', '--user',
        required=False,
        default='estenhl',
        help='Owner of the github repo'
    )
    parser.add_argument(
        '-r', '--repo',
        required=False,
        default='pyment-public',
        help='Name of the github repo'
    )

    args = parser.parse_args()

    response = upload_weights_to_github(
        filename=args.filename,
        token=args.token,
        user=args.user,
        repo=args.repo
    )

    print(response.text)
