import type { ComparisonMode, NucleotideFilterModel } from "./dbModels";

export default Object.freeze({
    defaultOrderBy: 'rmsdD',
    defaultDataSource: <NucleotideFilterModel["datasource"]>'allcontacts-f',
    defaultComparisonMode: <ComparisonMode>'union',
    /// default limit of displayed images in the gallery
    imgLimit: 100,
    /// hostname of server to use for images and parquet files when the webapp is running on localhost
    // debugHost: 'https://pairs.exyi.cz',
    debugHost: null,
    imgPath: '/pregen-img',
    fallbackImgPath: 'https://basepairs.datmos.org', // TODO
    tablesPath: '/tables',
    // disabledDataSources: [ ],
    disabledDataSources: [ "fr3d-nf", "fr3d-n", "fr3d", "allcontacts", "allcontacts-boundaries" ],

    parameterBoundariesUrls: {
        'googleTable': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTvEpcubhqyJoPTmL3wtq0677tdIRnkTghJcbPtflUdfvyzt4xovKJxBHvH2Y1VyaFSU5S2BZIimmSD/pub?gid=245758142&single=true&output=csv',
        fr3dData: 'parameter-boundaries.csv'
    },
    /// URL to basepairs.datmos.org/#s/X will be redirected to the full URL specified here
    shortLinks: {
        // TODO
        "s9pvei": "cWW-G-C/hb0_L=..3.4&hb0_DA=100..140&hb0_AA=100..140&hb1_L=..3.4&hb1_DA=100..140&hb1_AA=100..140&hb2_L=..3.4&hb2_DA=100..140&hb2_AA=100..140&baseline_ds=fr3d-f",
        "eris40": "tHH-A-G/hb0_L=..4&hb0_DA=100..150&hb0_AA=100..165&hb0_OOPA1=-25..35&hb0_OOPA2=-10..35&min_bond_length=..3.8&coplanarity_a=..40&coplanarity_edge_angle1=-10..25&coplanarity_edge_angle2=-10..30&coplanarity_shift1=-0.2..1.5&coplanarity_shift2=-0.3..1.3&baseline_ds=fr3d-f"
    },
    defaultBoundaries: [
        'googleTable',
        'fr3dData'
    ]
})
