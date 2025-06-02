const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const resetButton = document.getElementById('resetButton');
const startButton = document.getElementById('startButton');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = () => {
      imagePreview.src = reader.result;
      imagePreview.style.display = 'block';
      startButton.style.display = 'inline-block'; // Show the start button

      window.electronAPI?.sendImageData?.(reader.result); // Optional
    };
    reader.readAsDataURL(file);
  }
});

resetButton.addEventListener('click', () => {
  fileInput.value = '';
  imagePreview.src = '';
  imagePreview.style.display = 'none';
  startButton.style.display = 'none'; // Hide the start button
});