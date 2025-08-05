# Start from a stable Ubuntu LTS release
FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# Install basic dependencies, including curl and gnupg for adding repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
  sudo \
  git \
  zsh \
  curl \
  wget \
  gnupg \
  ca-certificates \
  less \
  procps \
  fzf \
  man-db \
  unzip \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Install Node.js and npm ---
# The official Ubuntu repos have an old version, so we use the NodeSource repository
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
RUN apt-get update && apt-get install -y nodejs

# --- Create a non-root user named "node" ---
# We create the user with UID/GID 1000, which is standard
RUN groupadd --gid 1000 node && \
    useradd --uid 1000 --gid 1000 --m --shell /bin/zsh node && \
    usermod -aG sudo node
# Set up passwordless sudo for the new user
RUN echo "node ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/node

# Persist bash history.
RUN mkdir /commandhistory && \
    touch /commandhistory/.bash_history && \
    chown -R node:node /commandhistory

# Set `DEVCONTAINER` environment variable
ENV DEVCONTAINER=true

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

WORKDIR /workspace

ARG GIT_DELTA_VERSION=0.18.2
# This command is robust; it will correctly report "amd64" inside the emulated container
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  sudo dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# --- Switch to the non-root user ---
USER node

# Install uv, the Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv and other local bin directories to the PATH
ENV PATH="/home/node/.cargo/bin:/home/node/.local/bin:/workspace/.venv/bin:${PATH}"

# Set the default shell to zsh
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude (npm was installed with nodejs)
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Copy and set up firewall script
COPY init-firewall.sh /usr/local/bin/
# Switch back to root temporarily to set permissions
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
  chmod 0440 /etc/sudoers.d/node-firewall
# Switch back to the default user
USER node