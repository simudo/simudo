#!/bin/bash
set -eo pipefail
_APT_PROXY_CONFPATH=/etc/apt/apt.conf.d/01proxy

_cmd_apt-update() {
    _cmd_configure-apt-proxy
    _cmd_apt-get -qq update
}
_cmd_configure-apt-proxy() {
    if [[ -n "$APT_HTTP_PROXY" ]]; then
        cat > $_APT_PROXY_CONFPATH <<-EOL
Acquire::HTTP::Proxy "${APT_HTTP_PROXY}";
Acquire::HTTPS::Proxy "false";
EOL
        # echo "Using host's apt proxy: APT_HTTP_PROXY=$APT_HTTP_PROXY" >&2
    else
        rm -f "$_APT_PROXY_CONFPATH"
        # echo "Deleted proxy configuration since APT_HTTP_PROXY is unset." >&2
    fi
}
_cmd_apt-get() {
    env DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confold" "$@"
}
_cmd_apt-upgrade() {
    _cmd_configure-apt-proxy
    _cmd_apt-get -qq update
    _cmd_apt-get dist-upgrade
}
_cmd_mergedir() {
    rsync -a "$1"/ "$2"/
    rm -rf "$1"
}
_cmd_clean() {
    apt-get -y clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    ( export APT_HTTP_PROXY=; _cmd_configure-apt-proxy; )
}

_name="_cmd_$1"
shift
"${_name}" "$@"
