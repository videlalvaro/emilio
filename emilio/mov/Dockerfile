# Dockerfile -- Reproducible build environment for emilio MOV-only binary
#
# Uses a 32-bit (i386) Debian base so the movfuscator, LCC, and the
# resulting binary are all native 32-bit -- no cross-compilation needed.
#
# Build:
#   docker build -t emilio-mov -f mov/Dockerfile mov/
#
# Extract the MOV-only binary:
#   docker run --rm -v $(pwd)/mov/build:/out emilio-mov
#
# Run with verification:
#   docker run --rm -v $(pwd)/mov/build:/out emilio-mov --verify
#
# The resulting emilio_mov binary will be copied to the mounted /out volume.

FROM i386/debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    git \
    patch \
    binutils \
    libc6-dev \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Copy source files
WORKDIR /build
COPY eml_mov.h eml_mov.c eml_tokenizer.c eml_test.c ./
COPY build_mov.sh verify_mov.sh test_mov.sh ./

RUN chmod +x build_mov.sh verify_mov.sh test_mov.sh

# Build movfuscator + compile emilio (done at image build time so the
# movfuscator toolchain doesn't need to be rebuilt every run).
RUN ./build_mov.sh --verify

# Default entrypoint: copy binary to /out, or just verify
ENTRYPOINT ["/bin/bash", "-c", "\
    if [ \"$1\" = \"--verify\" ]; then \
        ./verify_mov.sh ./build/emilio_mov; \
    elif [ -d /out ]; then \
        cp build/emilio_mov /out/emilio_mov && \
        echo 'Binary copied to /out/emilio_mov'; \
    else \
        echo 'Mount a volume to /out to extract the binary:'; \
        echo '  docker run --rm -v $(pwd)/mov/build:/out emilio-mov'; \
    fi \
", "--"]
