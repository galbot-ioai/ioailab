#!/usr/bin/env bash
#
# Start the GUI development shell and add serial-device access when present.
# Normal GUI development still works when no teleop serial device is attached.

set -euo pipefail

COMPOSE_FILE="${ioailab_COMPOSE_FILE:-docker/compose.yaml}"
GUI_SERVICE="${ioailab_GUI_SERVICE:-dev-gui}"
SERIAL_DEVICE_GLOBS=(/dev/ttyACM* /dev/ttyUSB*)
GP001_DETECT_ATTEMPTS="${GP001_DETECT_ATTEMPTS:-20}"
GP001_DETECT_SLEEP="${GP001_DETECT_SLEEP:-0.25}"

if [ -z "${ioailab_IMAGE:-}" ]; then
    ioailab_IMAGE_REPOSITORY="${ioailab_IMAGE_REPOSITORY:-ioailab}"
    if [ -z "${ioailab_IMAGE_TAG:-}" ]; then
        ioailab_VERSION_FILE="${ioailab_VERSION_FILE:-src/ioailab/__init__.py}"
        ioailab_IMAGE_TAG="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "${ioailab_VERSION_FILE}" | head -n 1)"
    fi
    if [ -n "${ioailab_IMAGE_TAG:-}" ]; then
        export ioailab_IMAGE="${ioailab_IMAGE_REPOSITORY}:${ioailab_IMAGE_TAG}"
    fi
fi

if [ "$#" -eq 0 ]; then
    set -- bash
fi

existing_serial_devices() {
    local requested="${GALBOT_GP001_DEVICE:-}"
    local candidate
    if [ -n "${requested}" ]; then
        if [ -e "${requested}" ]; then
            readlink -f "${requested}"
            return 0
        fi
        printf 'GP001 device %s not found.\n' "${requested}" >&2
        return 1
    fi

    for candidate in "${SERIAL_DEVICE_GLOBS[@]}"; do
        if [ -e "${candidate}" ]; then
            readlink -f "${candidate}"
        fi
    done | sort -u
}

wait_for_serial_devices() {
    local attempt
    local devices
    for ((attempt = 0; attempt <= GP001_DETECT_ATTEMPTS; attempt++)); do
        devices="$(existing_serial_devices || true)"
        if [ -n "${devices}" ]; then
            printf '%s\n' "${devices}"
            return 0
        fi
        sleep "${GP001_DETECT_SLEEP}"
    done
    return 1
}

run_gui_without_gp001() {
    if [ "${GP001_REQUIRED:-0}" = "1" ]; then
        printf 'GP001_REQUIRED=1 but no serial teleop device was found.\n' >&2
        printf 'Plug in the GP001 or set GALBOT_GP001_DEVICE=/dev/ttyACM* or /dev/ttyUSB*.\n' >&2
        exit 1
    fi

    printf 'GP001 not detected; GUI started without serial teleop device mapping.\n' >&2
    docker compose -f "${COMPOSE_FILE}" --profile gui run --rm "${GUI_SERVICE}" "$@"
}

serial_devices="$(wait_for_serial_devices || true)"
if [ -z "${serial_devices}" ]; then
    run_gui_without_gp001 "$@"
    exit 0
fi

override_file="$(mktemp -t ioailab-gui-serial.XXXXXX.yaml)"
trap 'rm -f "${override_file}"' EXIT

{
    printf 'services:\n'
    printf '  %s:\n' "${GUI_SERVICE}"
    printf '    devices:\n'
    while IFS= read -r device; do
        [ -n "${device}" ] || continue
        printf '      - %s:%s\n' "${device}" "${device}"
    done <<< "${serial_devices}"
    if [ -d /dev/serial/by-id ]; then
        printf '    volumes:\n'
        printf '      - /dev/serial/by-id:/dev/serial/by-id:ro\n'
    fi
} > "${override_file}"

printf 'Serial teleop device(s) detected; adding to %s:\n%s\n' "${GUI_SERVICE}" "${serial_devices}" >&2
docker compose -f "${COMPOSE_FILE}" -f "${override_file}" --profile gui run --rm "${GUI_SERVICE}" "$@"
