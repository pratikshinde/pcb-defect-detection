import { createApp } from './app';
import { getCv } from './cv/opencv';

const port = Number(process.env.PORT) || 3000;

async function main() {
  // Compile the OpenCV WASM now, so the first request does not pay for it (or race it).
  await getCv();
  createApp().listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${port}`);
  });
}

main().catch((err) => {
  console.error('Failed to start:', err);
  process.exit(1);
});
