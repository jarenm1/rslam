{
  description = "A development shell for C++ and Rust projects";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    {
      devShells.x86_64-linux.default =
        let
          pkgs = nixpkgs.legacyPackages.x86_64-linux;
        in
        pkgs.mkShell {
          buildInputs = [
            pkgs.gcc
            pkgs.gnumake
            pkgs.cmake
            pkgs.pkg-config

            pkgs.clang
            pkgs.llvm
            pkgs.libclang
            pkgs.libudev-zero
            pkgs.alsa-lib

            pkgs.xorg.libX11
            pkgs.xorg.libXcursor

            pkgs.rust-analyzer
            pkgs.rustfmt

            pkgs.opencv
            pkgs.onnxruntime

            pkgs.rustc
            pkgs.cargo

            pkgs.nil
            pkgs.nixfmt-classic
          ];

          shellHook = ''
            echo "Entering a Nix development shell."

            export PKG_CONFIG_PATH=${pkgs.opencv}/lib/pkgconfig:${pkgs.onnxruntime}/lib/pkgconfig:$PKG_CONFIG_PATH

            export LD_LIBRARY_PATH=${pkgs.opencv}/lib:${pkgs.onnxruntime}/lib:LD_LIBRARY_PATH

            export LLVM_CONFIG_PATH="${pkgs.llvm}/bin/llvm-config"
            export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

            echo "Complete"
          '';
        };
    };
}
