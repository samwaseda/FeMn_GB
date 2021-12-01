import numpy as np
from collections import defaultdict


def set_parameters(spx, n_cores=40, queue='cm', random=True):
    spx.set_convergence_precision(electronic_energy=1e-6)
    spx.set_kpoints(k_mesh_spacing=0.1)
    spx.set_encut(500)
    magmoms = 2 * np.ones(len(spx.structure))
    ind_Mn = spx.structure.select_index('Mn')
    if len(ind_Mn) == 1:
        magmoms[ind_Mn] *= -1
    if len(spx.structure.select_index('Mn')) > 1 and random:
        magmoms[ind_Mn] *= np.random.choice([1, -1], len(ind_Mn))
    spx.structure.set_initial_magnetic_moments(magmoms)
    spx.set_mixing_parameters(density_residual_scaling=0.1, spin_residual_scaling=0.1)
    spx.server.queue = queue
    spx.server.cores = n_cores
    spx.calc_minimize()
    return spx


class Bulk:
    def __init__(self, project):
        self.project = project

    def run_murnaghan(self):
        for element, a_0 in zip(['Fe', 'Mn'], [2.83, 2.8]):
            spx = self.project.create.job.Sphinx('bulk_{}'.format(element))
            spx.structure = self.project.create.structure.crystal(element, 'bcc', a_0)
            spx = set_parameters(spx, n_cores=4)
            murn = spx.create_job('Murnaghan', 'murn_{}'.format(element))
            if murn.status.initialized:
                murn.run()

    def get_lattice_constant(self):
        murn = self.project.inspect('murn_Fe')
        if murn is None:
            self.run_murnaghan()
            return None
        return murn['output/equilibrium_volume']**(1 / 3)

    def get_energy(self, element, n_repeat=3, n_Mn=1):
        if element == 'Fe':
            murn = self.project.inspect('murn_Fe')
            if murn is None:
                self.run_murnaghan()
                return None
            return murn['output/equilibrium_energy'] / 2
        elif element == 'Mn':
            murn = self.project.inspect('murn_sqs_{}_{}'.format(n_repeat, n_Mn))
            if murn is None:
                self.run_murnaghan()
                return None
            indices = murn['output/structure/indices']
            N_Fe = np.sum(np.array(murn['output/structure/species'])[indices] == 'Fe')
            coeff = np.polyfit(murn['output/volume'], murn['output/energy'], 3)
            return np.polyval(
                coeff, self.get_lattice_constant()**3 * len(indices) / 2
            ) - N_Fe * self.get_energy('Fe')
        else:
            raise ValueError(element, 'not recognized')


class GrainBoundary:
    def __init__(self, project):
        self.project = project
        self._energy_dict = None
        self._structure_dict = {}
        self.bulk = Bulk(project=self.project)

    def job_table(self, full_table=True):
        jt = self.project.job_table(full_table=full_table)
        return jt[jt.job.str.startswith('spx_gb_')]

    @property
    def job_names(self):
        return np.unique(['_'.join(j.split('_')[:-1]) for j in self.job_table().job])

    @property
    def energy_dict(self):
        if self._energy_dict is None:
            self._energy_dict = defaultdict(list)
            self.load_jobs()
        return self._energy_dict

    def load_jobs(self):
        jt = self.job_table()
        E_Fe = self.bulk.get_energy('Fe')
        for job_type in self.job_names:
            if any([
                s in list(jt[jt.job.str.startswith(job_type)].status) for s in ['submitted', 'running']
            ]):
                continue
            if any([k.startswith(job_type) for k in self._energy_dict.keys()]):
                continue
            for job_name in jt[jt.job.str.startswith(job_type)].job:
                spx = self.project.inspect(job_name)
                LL = spx['output/generic/cells'][-1]
                EE = E_Fe * len(spx['input/structure/indices'])
                self._energy_dict[job_type].append([
                    np.max(LL),
                    (spx['output/generic/energy_pot'][-1] - EE) / np.prod(np.diagonal(LL)) * np.max(LL) / 2
                ])
        for k, v in self._energy_dict.items():
            self._energy_dict[k] = np.array(v)[np.argsort(v, axis=0)[:, 0]]

    @staticmethod
    def _get_fit(x, y, order=3):
        coeff = np.polyfit(x, y, 3)
        c = np.polyder(coeff)
        return -0.5 * c[1] / c[0] - np.sqrt((c[1] / 2 / c[0])**2 - c[2] / c[0]), coeff

    @property
    def structures(self):
        if len(self._structure_dict) == 0:
            for k, v in self.energy_dict.items():
                L, coeff = self._get_fit(*v.T, order=3)
                structure = self.project.load('{}_0c0'.format(k)).structure
                cell = structure.cell.flatten()
                cell[cell.argmax()] = L
                structure.set_cell(cell.reshape(3, 3), scale_atoms=True)
                self._structure_dict[k] = structure.copy()
        return self._structure_dict

    def get_gb_energy(self, J_per_m2=False):
        results = {}
        for k, v in self.energy_dict.items():
            L, coeff = self._get_fit(*v.T, order=3)
            E = np.polyval(coeff, L)
            if J_per_m2:
                E *= 16.0219
            results[k] = E
        return results

    def get_angles(self):
        results = {}
        for job_name in self.energy_dict.keys():
            axis = [int(n) for n in job_name.split('_')[2].split('c')]
            sigma = int(job_name.split('_')[3])
            plane = []
            ss = ''
            for s in job_name.split('_')[4].replace('m', '-'):
                ss += s
                try:
                    plane.append(int(ss))
                    ss = ''
                except ValueError:
                    pass
            entry = self.project.create.structure.aimsgb.info(axis, sigma)[sigma]
            results[job_name] = np.array(entry['theta'])[
                np.all(np.array(entry['plane']) == plane, axis=-1).any(axis=-1)
            ].min()
        return results

    def get_segregation_energy(self):
        E_dict = {}
        jt = self.project.job_table()
        for job_name, structure in self.structures.items():
            E_lst = np.zeros(len(structure))
            equivalent_atoms = structure.get_symmetry(symprec=1.0e-2).arg_equivalent_atoms
            for atom_id in np.unique(equivalent_atoms):
                job_name_Mn = '{}_{}'.format(job_name.replace('gb', 'gbMn'), atom_id)
                if sum(jt.job == job_name_Mn) == 0:
                    spx = self.project.create.job.Sphinx('{}_{}'.format(job_name.replace('gb', 'gbMn'), atom_id))
                    spx.structure = structure.copy()
                    spx.structure[atom_id] = 'Mn'
                    set_parameters(spx)
                    spx.run()
                    continue
                if np.any([s == 'running' or s == 'submitted' for s in jt[jt.job == job_name_Mn].status]):
                    continue
                spx = self.project.inspect(job_name_Mn)
                gb_energy = self.get_gb_energy()[job_name] * structure.cell.diagonal().prod() / structure.cell.max()
                E_Fe = self.bulk.get_energy('Fe') * sum(
                    np.asarray(spx['input/structure/species'])[spx['input/structure/indices']] == 'Fe'
                )
                E = spx['output/generic/energy_pot'][-1] - 2 * gb_energy - E_Fe - self.bulk.get_energy('Mn', n_repeat=3)
                E_lst[equivalent_atoms == atom_id] = E
            if np.all(E_lst != 0):
                E_dict[job_name] = E_lst
        return E_dict
