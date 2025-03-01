#!/bin/bash 

# Exit on errors
set -e 

# Make sure we are one level above this directory
cd "$(dirname "$0")/.."

function build_flatnav() {
    # Check if wheelhouse directory exists. Only if it doesn't, build flatnav
    if [ ! -d "wheelhouse" ]; then
        echo "Building flatnav"
        build_flatnav_wheel
        ./cibuild.sh --current-version 3.12
    else
        echo "Flatnav wheel already exists"
    fi
}



function get_tag_name() {
    # Returns a string to be used as a docker tag revision.
    # If it's in a clean git repo, it returns the commit's short hash with branch name like ce37fd7-main
    # If the working tree is dirty, it returns something like main-ce37fd7-dirty-e52e78f86e575bd
    #     including the branch name, and a consistent hash of the uncommitted changes

    fail() {
        echo $1
        exit 1
    }

    if [[ ! -z "${OVERRIDE_GIT_TAG_NAME}" ]]; then
        echo $OVERRIDE_GIT_TAG_NAME
        exit 0
    fi

    # Figure out which SHA utility exists on this machine.
    HASH_FUNCTION=sha1sum
    which $HASH_FUNCTION > /dev/null || HASH_FUNCTION=shasum
    which $HASH_FUNCTION > /dev/null || fail "Can't find SHA utility"

    # Try to get current branch out of GITHUB_REF for CI
    # The ##*/ deletes everything up to /
    CURRENT_BRANCH=${GITHUB_REF##*/}
    # Now generate the short commit
    CURRENT_COMMIT=$(echo $GITHUB_SHA | cut -c -9)

    # If we're not running in CI, GITHUB_REF and GITHUB_SHA won't be set.
    # In this case, figure them out from our git repository
    # (If we do this during github CI, we get a useless unique commit on the "merge" branch.)
    # When infering CURRENT_BRANCH, convert '/'s to '-'s, since '/' is not allowed in docker tags but
    # is part of common git branch naming formats e.g. "feature/branch-name" or "user/branch-name"
    CURRENT_BRANCH=${CURRENT_BRANCH:-$(git rev-parse --abbrev-ref HEAD | sed -e 's/\//-/g')}
    CURRENT_COMMIT=${CURRENT_COMMIT:-$(git rev-parse --short=9 HEAD)}

    if [[ -z "$(git status --porcelain)" ]] || [[ "${CI}" = true ]]; then
        # Working tree is clean
        echo "${CURRENT_COMMIT}-${CURRENT_BRANCH}"
    else
        # Working tree is dirty.
        HASH=$(echo $(git diff && git status) | ${HASH_FUNCTION} | cut -c -15)
        echo "${CURRENT_BRANCH}-${CURRENT_COMMIT}-dirty-${HASH}"
    fi
}

# Get the tag name
TAG_NAME=$(get_tag_name)

build_flatnav
cp wheelhouse/* program_discovery/wheelhouse/

# Print commands and their arguments as they are executed
set -x

DATA_DIR=${DATA_DIR:-$(pwd)/data}

# Directory for storing metrics and plots. 
METRICS_DIR=${METRICS_DIR:-$(pwd)/metrics}
CONTAINER_NAME=${CONTAINER_NAME:-prog-discovery-runner}

echo "Building docker image with tag name: $TAG_NAME"

# If data directory doesn't exist, exit 
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR"
    exit 1
fi
mkdir -p $METRICS_DIR

# Clean up existing docker images matching "flatnav" if any 
docker rmi -f $(docker images --filter=reference="program-discovery" -q) &> /dev/null || true

cd program_discovery
docker build --tag program-discovery:$TAG_NAME -f Dockerfile .


# Run the container and mount the data/ directory as volume to /root/data
docker run \
        --name $CONTAINER_NAME \
        -it \
        --rm program-discovery:$TAG_NAME