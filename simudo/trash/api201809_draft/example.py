
pdata = ProblemData()

pdata.spatial = spatial = Spatial()

pdata.optical = optical = Optical()

pdata.PDD = pdd = PoissonDriftDiffusion(pdata=pdata)

pdd.poisson = PoissonProblem(PDD=pdd)

if solve_for_band_density: # false for pre-PDD solver steps, e.g. local
                           # neutrality and Poisson-only thermal equilibrium
    xConductionBand = ZeroQflNondegenerateConductionBand
    xValenceBand = ZeroQflNondegenerateValenceBand
    xIntermediateBand = ZeroQflIntermediateBand
else:
    xConductionBand = MixedQflNondegenerateConductionBand
    xValenceBand = MixedQflNondegenerateValenceBand
    xIntermediateBand = MixedQflIntermediateBand

pdd.bands.add(xConductionBand(PDD=pdd))
pdd.bands.add(xValenceBand(PDD=pdd))
pdd.bands.add(xIntermediateBandx(PDD=pdd))


pdd.poisson.bcs['V']['p_contact'] = 0.5 * u.V
pdd.poisson.bcs['V']['n_contact'] = 0.8 * u.V
pdd.poisson.bcs['E']['exterior_minus_contacts'] = 0 * u('V/cm')

for b in pdd.bands.values():
    b.bcs['j']['exterior_minus_contacts'] = 0.0 * u('mA/cm^2')

# extra configuration for BCs
CB = pdd.bands['CB']
CB.bcs['u']['contacts'] = SurfaceRecombinationVelocityBC(CB, value=1e7*u('cm/s'))

VB = pdd.bands['VB']
VB.bcs['u']['contacts'] = SurfaceRecombinationVelocityBC(VB, value=1e7*u('cm/s'))

eops = pdd.electro_optical_processes

eops.append(SimpleLinearAbsorption(
    dst_band=VB, src_band=CB))

eops.append(CrossSectionAbsorption(
    dst_band=CB, src_band=IB, density_expression=IB.u))

eops.append(CrossSectionAbsorption(
    dst_band=IB, src_band=VB, density_expression=(IB.N0-IB.u)))

# material config

```
class MyMaterial():
    def get_CB__band_edge(self):
        return 1.12*u.eV
    ...

class ModifiedMaterial(MyMaterial):
    def get_CB__band_edge(self):
        return super().get_CB__band_edge() + 0.1*u.eV

    ## TEMPERATURE DEP!
```


spatial.add_rule(region='silicon', dim=2, data={
    # 'spatial_property': constant_or_dolfin_expression,
    'CB/band_edge': 1.12*u.eV,
    'VB/band_edge': 0.0*u.eV,
    'CB/density_of_states': 3.2e19*u('cm^-3'),
    'VB/density_of_states': 1.8e19*u('cm^-3'),
    'CB/mobility': 1400*u('cm^2 V^-1 s^-1'),
    'VB/mobility':  450*u('cm^2 V^-1 s^-1'),
    'permittivity': 11.7*vacuum_permittivity,
})

spatial.add_rule(region='everywhere', dim=2, data={
    'IB/N0': 0*u('cm^-3'),
})

# later rules override earlier ones
spatial.add_rule(region='p_type', dim=2, data={'doping': -1e17*u('cm^-3')})
spatial.add_rule(region='n_type', dim=2, data={'doping': +1e17*u('cm^-3')})
spatial.add_rule(region='IB_region', dim=2, data={
    'IB/N0': +1e17*u('cm^-3'),
    'IB/mobility': ...,
    'CB/mobility': ....
    'VB/mobility': ...})



# for completeness' sake, here's what mesh generation code *currently*
# looks like, along with the code defining the regions and boundary
# regions.

class Bob(ConstructionHelper):
    def user_define_mshr_regions(self):
        '''must return `{region_name: mshr_domain}`
        `regions['domain']` is overall domain'''
        def r(x0, y0, x1, y1):
            return mshr.Rectangle(dolfin.Point(x0, y0), dolfin.Point(x1, y1))

        p = self.p

        x_pn = p.p_length

        h = p.height

        d = {}
        d['pSi'] = r(0, 0, x_pn, h)
        d['overmesh_pn'] = r(x_pn - p.overmesh_pn[0], 0,
                             x_pn + p.overmesh_pn[1], h)
        d['nSi'] = r(x_pn, 0, p.length, h)
        d['everywhere'] = r(0, 0, p.length, h)

        for name, (x0, x1) in p.extra_regions.items():
            d[name] = r(x0, 0, x1, h)

        if p.near_p_contact_length:
            d['near_p_contact'] = r(0, 0, p.near_p_contact_length, h)

        return d

    def user_extra_definitions(self):
        '''override me'''

        R = self.cell_regions
        F = self.facet_regions
        fc = self.facets

        R['Si'] = R['pSi'] | R['nSi']

        F.update(
            p_contact=fc.boundary(R['left'], R['pSi']),
            n_contact=fc.boundary(R['nSi'], R['right']))

        F['top_bottom'] = fc.boundary(R['everywhere'], R['top'] | R['bottom'])
        F['contacts'] = F['p_contact'] | F['n_contact']

    def user_refinement(self):
        R = self.cell_regions
        p = self.p

        def refine_in_x(subdomains, threshold, skew=20.0):
            pred = mep.DirectionalEdgeLengthPredicate(
                np.array([1.0, 0.0, 0.0]), threshold)
            trans = LinearTransform([1.0, 1/skew])
            trans.transform(self.mesh)
            self.refine_subdomains(subdomains, pred)
            trans.untransform(self.mesh)

        for k, (threshold, skew) in p.extra_refine_in_x.items():
            refine_in_x(R[k], threshold, skew)

        if 'near_p_contact' in R:
            refine_in_x(R['near_p_contact'], p.height/8, skew=1.0)

