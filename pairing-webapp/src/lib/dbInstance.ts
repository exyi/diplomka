
import { AsyncDuckDB, AsyncDuckDBConnection, DuckDBDataProtocol } from '@duckdb/duckdb-wasm';
import metadata from '$lib/metadata.ts';
import { initDB } from '$lib/duckdb'
import { normalizePairFamily, compareFamilies } from '$lib/pairing'
// let db: any = null;
let conn: AsyncDuckDBConnection | null = null

export const parquetFiles = {
}
export const pairTypes = metadata.map(m => m.pair_type)
export const pairFamilies = [...new Set(pairTypes.map(t => normalizePairFamily(t[0])))]

pairFamilies.sort(compareFamilies)
const cacheBuster = '?v=2'

for (const pairMeta of metadata) {
  const [family, bases] = pairMeta.pair_type
  if (pairMeta.count != 0) {
    parquetFiles[`${family}-${bases}`] = `${family}-${bases}.parquet${cacheBuster}`
    parquetFiles[`${normalizePairFamily(family)}-${bases}`] = `${family}-${bases}.parquet${cacheBuster}`
    parquetFiles[`${family}-${bases}-filtered`] = `${family}-${bases}-filtered.parquet${cacheBuster}`
    parquetFiles[`${normalizePairFamily(family)}-${bases}-filtered`] = `${family}-${bases}-filtered.parquet${cacheBuster}`
    parquetFiles[`n${family}-${bases}`] = `n${family}-${bases}.parquet${cacheBuster}`
    parquetFiles[`${family}-${bases}_n`] = `n${family}-${bases}.parquet${cacheBuster}`
  }
}
export const host = window.location.hostname.match(/(^|[.])localhost$/) ? 'localhost' : window.location.hostname
export const fileBase = (new URL('tables/', host == 'localhost' ? 'https://pairs.exyi.cz/' : document.baseURI)).href

export function getConnectionSync(): AsyncDuckDBConnection {
  if (!conn)
    throw new Error("DuckDB connection not initialized")
  return conn
}

export async function connect(): Promise<AsyncDuckDBConnection> {
  if (conn) {
    return conn
  }
  console.log("LOADING DB")
  const db = await initDB();
  for (const [name, url] of Object.entries(parquetFiles)) {
    await db.registerFileURL(name, `${fileBase}${url}`, DuckDBDataProtocol.HTTP, false);
  }
  conn = await db.connect();

  // console.log("PREPARING VIEWS")
  // const existingTables = await db.getTableNames(conn, )
  // for (const [name, url] of Object.entries(parquetFiles)) {
  //   await conn.query(`CREATE OR REPLACE VIEW '${name}' AS SELECT * FROM parquet_scan('${name}')`);
  // }
  window["duckdbconn"] = conn
  return conn;
}
