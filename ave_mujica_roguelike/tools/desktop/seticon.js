// Replace the exe icon and version strings without Windows tools (pure-JS PE editing).
const fs = require('fs'); const ResEdit = require('resedit');
const [exe, ico] = process.argv.slice(2);
const pe = ResEdit.NtExecutable.from(fs.readFileSync(exe), { ignoreCert: true });
const res = ResEdit.NtExecutableResource.from(pe);
const icon = ResEdit.Data.IconFile.from(fs.readFileSync(ico));
const groups = ResEdit.Resource.IconGroupEntry.fromEntries(res.entries);
const gid = groups.length ? groups[0].id : 1, lang = groups.length ? groups[0].lang : 1033;
ResEdit.Resource.IconGroupEntry.replaceIconsForResource(res.entries, gid, lang, icon.icons.map(i => i.data));
const vi = ResEdit.Resource.VersionInfo.fromEntries(res.entries)[0];
if (vi) {
  const l = { lang: 1033, codepage: 1200 };
  vi.setStringValues(l, { ProductName: 'Ave Mujica', FileDescription: 'Ave Mujica · 假面回廊', CompanyName: 'fan-made', OriginalFilename: 'AveMujica.exe', InternalName: 'AveMujica', LegalCopyright: 'Fan-made, non-commercial' });
  vi.setFileVersion(1, 0, 0, 0, 1033); vi.setProductVersion(1, 0, 0, 0, 1033); vi.outputToResourceEntries(res.entries);
}
res.outputResource(pe); fs.writeFileSync(exe, Buffer.from(pe.generate()));
console.log('icon + version info set');
