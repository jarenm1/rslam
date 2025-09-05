{
  description = "A Nix flake for a general Rust development environment";

  # --- Flake Inputs ---
  # These are the dependencies of our flake. We pin them to specific versions
  # for reproducibility.
  inputs = {
    # The main Nix package set
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # A utility library to make flakes easier to write
    flake-utils.url = "github:numtide/flake-utils";

    # An overlay that provides up-to-date Rust toolchains
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  # --- Flake Outputs ---
  # This function defines what the flake provides, such as packages, shells, etc.
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    # Use flake-utils to generate outputs for common systems (x86_64-linux, aarch64-linux, etc.)
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # --- Overlays ---
        # Overlays allow us to add or override packages in nixpkgs.
        # Here, we add the rust-overlay to get the latest Rust toolchains.
        overlays = [ (import rust-overlay) ];

        # --- Package Set ---
        # We import nixpkgs for the specified system and apply our overlays.
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
      {
        # --- Development Shell ---
        # This defines the 'nix develop' shell environment.
        devShells.default = pkgs.mkShell {
          # --- Packages ---
          # These are the tools and libraries available in the shell.
          packages = with pkgs; [
            # This provides the core Rust toolchain (rustc, cargo, rustfmt, etc.)
            # from the rust-overlay. You can easily change the version here.
            rust-bin.stable.latest.default

            # The official language server for Rust. Essential for IDE features
            # like autocompletion, go-to-definition, and diagnostics.
            rust-analyzer

            # A language server for the Nix language itself, which is very helpful
            # for editing this flake.nix file.
            nil

            # --- Common Build Dependencies ---
            # Many Rust crates need to link against C libraries. These packages
            # are common requirements.

            # A tool to help find system libraries
            pkg-config

            # The OpenSSL library, required by many web and crypto crates
            openssl

            # A C/C++ compiler toolchain
            clang
            llvmPackages.bintools
            llvmPackages.llvm
            llvmPackages.libclang

            # project cpp packages
            opencv
          ];

          # --- Environment Variables ---
          shellHook = ''
            export RUST_SRC_PATH="${pkgs.rust-bin.stable."1.79.0".rust-src}/lib/rustlib/src/rust/library"
            export LLVM_CONFIG_PATH="${pkgs.llvmPackages.llvm}/bin/llvm-config"
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
          '';
        };
      }
    );
}
