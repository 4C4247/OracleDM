const fs = require('fs');
const path = require('path');
const { Resvg } = require('@resvg/resvg-js');

const frame = path.resolve(__dirname, '..', '..', 'paper', 'frame');
const svg = fs.readFileSync(path.join(frame, '_svg', 'full.svg'), 'utf8');
const resvg = new Resvg(svg, {
  fitTo: { mode: 'width', value: 5280 },
  background: '#ffffff',
  font: { loadSystemFonts: true, defaultFontFamily: 'Microsoft YaHei' },
});
fs.writeFileSync(path.join(frame, 'framework_full.png'), resvg.render().asPng());
console.log('done');
