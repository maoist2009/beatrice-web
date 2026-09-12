/**
 * Minimal TOML parser — enough for `beatrice_paraphernalia_*.toml`
 * ([model], [voice.N], [voice.N.portrait], strings incl. triple-quoted, numbers, booleans).
 */
export type TomlValue = string | number | boolean | TomlTable | TomlValue[];
export interface TomlTable { [k: string]: TomlValue }

export function parseToml(src: string): TomlTable {
  const root: TomlTable = {};
  let cur: TomlTable = root;
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  const getTable = (path: string[]): TomlTable => {
    let t = root;
    for (const p of path) {
      if (typeof t[p] !== "object" || Array.isArray(t[p])) t[p] = {};
      t = t[p] as TomlTable;
    }
    return t;
  };
  const splitKey = (k: string) => k.split(".").map((s) => s.trim().replace(/^"(.*)"$/, "$1"));
  while (i < lines.length) {
    let line = lines[i].trim();
    if (!line || line.startsWith("#")) { i++; continue; }
    const th = /^\[([^\]]+)\]\s*(#.*)?$/.exec(line);
    if (th) { cur = getTable(splitKey(th[1])); i++; continue; }
    const eq = line.indexOf("=");
    if (eq < 0) { i++; continue; }
    const key = splitKey(line.slice(0, eq));
    let rhs = line.slice(eq + 1).trim();
    let value: TomlValue;
    if (rhs.startsWith('"""') || rhs.startsWith("'''")) {
      const q = rhs.slice(0, 3);
      let body = rhs.slice(3);
      let end = body.indexOf(q);
      if (end < 0) {
        const parts = [body];
        i++;
        while (i < lines.length) {
          const idx = lines[i].indexOf(q);
          if (idx >= 0) { parts.push(lines[i].slice(0, idx)); break; }
          parts.push(lines[i]); i++;
        }
        body = parts.join("\n");
        if (body.startsWith("\n")) body = body.slice(1);
        value = q === '"""' ? unescapeBasic(body) : body;
      } else {
        value = q === '"""' ? unescapeBasic(body.slice(0, end)) : body.slice(0, end);
      }
    } else if (rhs.startsWith('"')) {
      let j = 1; let out = "";
      while (j < rhs.length && rhs[j] !== '"') { if (rhs[j] === "\\") { out += rhs[j] + rhs[j + 1]; j += 2; } else out += rhs[j++]; }
      value = unescapeBasic(out);
    } else if (rhs.startsWith("'")) {
      value = rhs.slice(1, rhs.indexOf("'", 1));
    } else {
      rhs = rhs.replace(/\s+#.*$/, "");
      if (rhs === "true") value = true;
      else if (rhs === "false") value = false;
      else if (/^[+-]?(\d[\d_]*)(\.\d+)?([eE][+-]?\d+)?$/.test(rhs)) value = Number(rhs.replace(/_/g, ""));
      else value = rhs;
    }
    let t = cur;
    for (let k = 0; k < key.length - 1; k++) {
      if (typeof t[key[k]] !== "object") t[key[k]] = {};
      t = t[key[k]] as TomlTable;
    }
    t[key[key.length - 1]] = value;
    i++;
  }
  return root;
}

function unescapeBasic(s: string): string {
  return s.replace(/\\(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)/g, (_m, g: string) => {
    switch (g[0]) {
      case "n": return "\n"; case "t": return "\t"; case "r": return "\r";
      case '"': return '"'; case "\\": return "\\"; case "b": return "\b"; case "f": return "\f";
      case "u": case "U": return String.fromCodePoint(parseInt(g.slice(1), 16));
      default: return g;
    }
  });
}
