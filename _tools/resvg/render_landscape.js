const fs = require('fs');
const path = require('path');
const { Resvg } = require('@resvg/resvg-js');

const frame = path.resolve(__dirname, '..', '..', 'paper', 'frame');
const svgDir = path.join(frame, '_svg');
const jobs = [
  ['landscape_cn', 'framework_landscape_cn', 'Microsoft YaHei'],
  ['landscape_en', 'framework_landscape_en', 'Arial'],
];

for (const [base, out, font] of jobs) {
  const svg = fs.readFileSync(path.join(svgDir, base + '.svg'), 'utf8');
  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: 4960 },
    background: '#ffffff',
    font: { loadSystemFonts: true, defaultFontFamily: font },
  });
  const png = resvg.render().asPng();
  fs.writeFileSync(path.join(frame, out + '.png'), png);
  console.log('saved', out);
}
console.log('done');
