function copyInstall() {
    const command =
        "git clone https://github.com/tlplayer/Severian.git";

    navigator.clipboard.writeText(command);

    const button = document.getElementById("copy-button");

    button.textContent = "Copied";

    setTimeout(() => {
        button.textContent = "Copy";
    }, 1500);
}
