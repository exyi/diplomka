import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18'
import { createPluginUI } from 'molstar/lib/mol-plugin-ui'
import type { NucleotideId, PairId } from '$lib/pairing'
import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context'
import { DefaultPluginUISpec, type PluginUISpec } from 'molstar/lib/mol-plugin-ui/spec'
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { PluginConfig } from 'molstar/lib/mol-plugin/config'
import "molstar/lib/mol-plugin-ui/skin/light.scss"
import { PluginSpec } from 'molstar/lib/mol-plugin/spec'
import { AssemblySymmetry } from 'molstar/lib/extensions/assembly-symmetry'
import { DownloadStructure, PdbDownloadProvider } from 'molstar/lib/mol-plugin-state/actions/structure';
import { DownloadDensity } from 'molstar/lib/mol-plugin-state/actions/volume';
import { StructureRepresentationPresetProvider, presetStaticComponent } from 'molstar/lib/mol-plugin-state/builder/structure/representation-preset'
import { StateObjectRef, StateObjectSelector } from 'molstar/lib/mol-state'
import { Material } from 'molstar/lib/mol-util/material'
import { ResidueQuery, StructureSelectionCategory, StructureSelectionQuery } from 'molstar/lib/mol-plugin-state/helpers/structure-selection-query'
import { transpiler as pymolTranspiler } from 'molstar/lib/mol-script/transpilers/pymol/parser'

import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Transparency } from 'molstar/lib/mol-theme/transparency'
import { StructureRepresentation3D, TransparencyStructureRepresentation3DFromBundle, TransparencyStructureRepresentation3DFromScript } from 'molstar/lib/mol-plugin-state/transforms/representation'
import { setStructureTransparency } from 'molstar/lib/mol-plugin-state/helpers/structure-transparency'
import { type ElementIndex, StructureSelection } from 'molstar/lib/mol-model/structure'
import { Bundle } from 'molstar/lib/mol-model/structure/structure/element/bundle'
import { RuntimeContext, Task } from 'molstar/lib/mol-task'
import { SymmetryOperator } from 'molstar/lib/mol-math/geometry'
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra'
import { Loci } from 'molstar/lib/mol-model/structure/structure/element/loci'
import { Script } from 'molstar/lib/mol-script/script'

export async function createContext(element: HTMLElement): Promise<PluginUIContext> {

    const defaultSpec = DefaultPluginUISpec()
    const spec: PluginUISpec = {
        ...defaultSpec,
        behaviors: [
            ...defaultSpec.behaviors,
            PluginSpec.Behavior(AssemblySymmetry)
        ],
        layout: {
            initial: {
                isExpanded: false,
                showControls: false,
                controlsDisplay: 'portrait'
            }
        },
        components: {
            ...defaultSpec.components,
        },
        config: [
            // [PluginConfig.General.DisableAntialiasing, o.disableAntialiasing],
            // [PluginConfig.General.PixelScale, o.pixelScale],
            // [PluginConfig.General.PickScale, o.pickScale],
            // [PluginConfig.General.Transparency, PluginConfig.General.Transparency.defaultValue],
            // [PluginConfig.General.PreferWebGl1, o.preferWebgl1],
            // [PluginConfig.General.AllowMajorPerformanceCaveat, o.allowMajorPerformanceCaveat],
            // [PluginConfig.General.PowerPreference, o.powerPreference],
            // [PluginConfig.Viewport.ShowExpand, o.viewportShowExpand],
            // [PluginConfig.Viewport.ShowControls, o.viewportShowControls],
            // [PluginConfig.Viewport.ShowSettings, o.viewportShowSettings],
            // [PluginConfig.Viewport.ShowSelectionMode, o.viewportShowSelectionMode],
            [PluginConfig.Viewport.ShowAnimation, false],
            [PluginConfig.Viewport.ShowTrajectoryControls, false],
            // [PluginConfig.State.DefaultServer, o.pluginStateServer],
            // [PluginConfig.State.CurrentServer, o.pluginStateServer],
            [PluginConfig.VolumeStreaming.DefaultServer, 'https://www.ebi.ac.uk/pdbe/densities'],
            [PluginConfig.VolumeStreaming.Enabled, true],
            [PluginConfig.Download.DefaultPdbProvider, "pdbe"],
            [PluginConfig.Download.DefaultEmdbProvider, "pdbe"],
            // [PluginConfig.Structure.DefaultRepresentationPreset, ViewerAutoPreset.id],
            // [PluginConfig.Structure.SaccharideCompIdMapType, o.saccharideCompIdMapType],
            // [VolsegVolumeServerConfig.DefaultServer, o.volumesAndSegmentationsDefaultServer],
            // [AssemblySymmetryConfig.DefaultServerType, o.rcsbAssemblySymmetryDefaultServerType],
            // [AssemblySymmetryConfig.DefaultServerUrl, o.rcsbAssemblySymmetryDefaultServerUrl],
            // [AssemblySymmetryConfig.ApplyColors, o.rcsbAssemblySymmetryApplyColors],
        ]
    }


    const ui = createPluginUI({
        target: element,
        render: renderReact18,
        spec
    })

    return ui
}

function nucleotideSelector(nt: NucleotideId) {

    // pymolTranspiler(`resi ${nt.resnum} and chain ${nt.chain}`)
    // return MS.struct.type.labelResidueId([ nt.chain, nt.symop ?? '1_555', nt.resnum, nt.inscode ])
    const intersectAll = [
        pymolTranspiler(`chain ${nt.chain}`),
        pymolTranspiler(`resi ${nt.resnum}`)
    ]
    if (nt.altloc) {
        intersectAll.push(pymolTranspiler(`alt ${nt.altloc}`))
    }
    return intersectAll.reduce((a, b) => MS.struct.modifier.intersectBy({ 0: a, by: b }))
}

function pairSelector(s: PairId):StructureSelectionQuery {
    const nt1 = nucleotideSelector(s.nt1)
    const nt2 = nucleotideSelector(s.nt2)
    const expr = MS.struct.combinator.merge([ nt1, nt2 ])
    // function resSele(r: NucleotideId) {
    //     let nt = String(r.resnum).replace("-", "\\-")
    //     if (r.inscode) {
    //         nt += String(r.inscode)
    //     }

    //     const alt = r.altloc ? ` and alt ${r.altloc}` : ""
    //     return `resi ${nt}${alt}`
    // }

    // const pymol = `((chain ${s.nt1.chain} and ${resSele(s.nt1)}) or (chain ${s.nt2.chain} and ${resSele(s.nt2)}))`
    // const expr = pymolTranspiler(pymol)

    return StructureSelectionQuery(
        `pair-${s.pairingType[0]}-${s.pairingType[1]}`, 
        expr,
        { category: StructureSelectionCategory.Misc }
    )
}


function meanCoordinates(c: SymmetryOperator.ArrayMapping<ElementIndex>[]): { center: Vec3, r: number } {
    const n = c.reduce((a, b) => a + b.coordinates.x.length, 0)
    let x = 0
    let y = 0
    let z = 0
    for (const m of c) {
        for (let i = 0, _i = m.coordinates.x.length; i < _i; i++) {
            x += m.x(i as ElementIndex)
            y += m.y(i as ElementIndex)
            z += m.z(i as ElementIndex)
        }
    }
    const center = Vec3.create(x / n, y / n, z / n)
    let r = 0
    const s = Vec3.create(0, 0, 0)
    for (const m of c) {
        for (let i = 0, _i = m.coordinates.x.length; i < _i; i++) {
            r = Math.max(Vec3.squaredDistance(m.position(i as ElementIndex, s), center))
        }
    }
    return {
        center,
        r: Math.sqrt(r)
    }
}

const StructurePreset = (pairSelector: StructureSelectionQuery) => StructureRepresentationPresetProvider({
    id: 'preset-structure',
    display: { name: 'Structure' },
    params: () => StructureRepresentationPresetProvider.CommonParams,
    async apply(ref, params, plugin) {
        const structureCell = StateObjectRef.resolveAndCheck(plugin.state.data, ref);
        if (!structureCell) return {};

        // ResidueQuery(

        const notPair = StructureSelectionQuery(
            'not-pair',
            MS.struct.modifier.exceptBy({
                0: MS.struct.generator.all(),
                by: pairSelector.expression
            }),
            { category: StructureSelectionCategory.NucleicBase })

        const components = {
            all: await plugin.builders.structure.tryCreateComponentFromSelection(structureCell, notPair, 'all'),
            // all: await plugin.builders.structure.tryCreateComponentStatic(structureCell, 'all'),
            pair: await plugin.builders.structure.tryCreateComponentFromSelection(structureCell, pairSelector, 'pair')
        };

        const { update, builder, typeParams } = StructureRepresentationPresetProvider.reprBuilder(plugin, params);
        const CustomMaterial = Material({ roughness: 0.2, metalness: 0 });
        // Transparency
        // update.to(components.all).apply(TransparencyStructureRepresentation3DFromScript, {  }, {  })
        const representations = {
            pair: builder.buildRepresentation(update, components.pair, { type: 'ball-and-stick', typeParams: { ...typeParams, material: CustomMaterial }, color: 'element-symbol', colorParams: { palette: (plugin.customState as any).colorPalette } }, { tag: 'pair' }),

            all: builder.buildRepresentation(update, components.all, { type: 'cartoon', typeParams: { ...typeParams, material: CustomMaterial }, color: 'chain-id', colorParams: { palette: (plugin.customState as any).colorPalette } }, { tag: 'polymer' }),
        };

        // notPair.getSelection(plugin, null, null).then((a: StructureSelection) => a.kind)

        const t_ = Transparency.ofBundle([
            // ...representations.all.cell.params.values.layers,
            { value: 0.5, bundle: Bundle.fromSubStructure(structureCell.obj.data, components.all.obj.data) }
        ], components.all.obj.data)
        const t = Transparency.filter(Transparency.merge(t_), components.all.obj.data)
        update.to(representations.all).apply(TransparencyStructureRepresentation3DFromBundle, Transparency.toBundle(t as any), { tags: 'transparency-controls' })

        // setStructureTransparency(plugin, representations.all, 0.5, null)

        await update.commit({ revertOnError: true });
        // await shinyStyle(plugin);
        plugin.managers.interactivity.setProps({ granularity: 'residue' });


        const pairLoci = StructureSelection.toLociWithSourceUnits(Script.getStructureSelection(pairSelector.expression, structureCell.obj.data))
        console.log(pairLoci)
        plugin.managers.structure.focus.setFromLoci(pairLoci)
        plugin.managers.interactivity.lociSelects.select({ loci: pairLoci })
        plugin.managers.camera.focusLoci(pairLoci, {durationMs: 1000, minRadius: 10 })

        // console.log(components.pair.obj.data.units.map(u => u.conformation))
        // const center = meanCoordinates(components.pair.obj.data.units.map(u => u.conformation))
        // plugin.canvas3d.camera.focus(center.center, center.r, 1000)
        // PluginCommands.Camera.Focus(plugin, { center: center.center, radius: 10, durationMs: 1000 })

        return { components, representations };
    }
});

async function fetchStructure(cx: PluginUIContext, pdbid: string) {
    const params = DownloadStructure.createDefaultParams(cx.state.data.root.obj, cx);
    const provider = cx.config.get(PluginConfig.Download.DefaultPdbProvider);
    const representationParams: StructureRepresentationPresetProvider.CommonParams | null = null
    
    await cx.runTask(cx.state.data.applyAction(DownloadStructure, {
        source: {
            name: 'pdb' as const,
            params: {
                provider: {
                    id: pdbid,
                    server: {
                        name: provider,
                        params: PdbDownloadProvider[provider].defaultValue as any
                    }
                },
                options: { ...params.source.params.options, representationParams },
            }
        }
    }));
    // const url = `${cx.config.get(PluginConfig.State.CurrentServer)}/get/${pdbid}`
    // await PluginCommands.State.Snapshots.Fetch(cx, { url })
}

export class MolStarViewer {
    constructor(private cx: PluginUIContext) { }
    
    currentPairId: PairId | null = null
    public async loadPair(pairId: PairId) {
        if (this.currentPairId?.nt1.pdbid !== pairId.nt1.pdbid) {
            await fetchStructure(this.cx, pairId.nt1.pdbid)
        }

        const mng = this.cx.managers.structure
        const struct = mng.hierarchy.current.structures
        const pairSel = pairSelector(pairId)
        mng.component.applyPreset(struct, StructurePreset(pairSel))
        // PluginCommands.State.SetCurrentObject()
    }
}


export async function createMolStar(container: HTMLElement): Promise<MolStarViewer> {
    const ui = await createContext(container)
    const v = new MolStarViewer(ui)
    return v
}
