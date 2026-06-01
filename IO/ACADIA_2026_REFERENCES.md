# Drop Ceiling — Candidate References (ACM Digital Library)

*Verified against the ACM Digital Library full-text collection (`dl.acm.org`) on 2026-05-31.
Each entry lists the venue, year, pages, and DOI exactly as shown in ACM's own catalogue.
Grouped by the four areas requested. These are **candidates** — read abstracts before
citing; relevance notes say why each is plausible for Drop Ceiling. ACADIA proceedings are
indexed in CumInCAD, but these ACM sources are the requested verifiable anchors.*

> **Framing note on "AR".** Drop Ceiling is light/illumination in physical space driven by
> camera tracking, with no headset and no screen-space overlay. The precise ACM term for
> this is **Spatial Augmented Reality (SAR)** / projection-based AR — digital response
> registered to real surfaces, viewable without worn devices. Framing the project as SAR
> (rather than HMD/optical-see-through AR) is both more accurate and better supported by
> the literature below. The fiducial-marker calibration that registers the cameras to the
> floor is itself a classic AR registration technique, which ties §1 and §3 together.

---

## 1. ArUco / fiducial markers, homography & multi-camera registration

The installation calibrates two camera feeds to a common floor plane using ArUco markers
and homography/pose estimation, then fuses detections into one world coordinate frame.

1. **Artificial Markers: A Comprehensive Systematic Review and Design Framework.**
   Benedito Ribeiro Neto, Bianchi Meiguins, Tiago Araújo, Carlos dos Santos.
   *ACM Computing Surveys (CSUR)* 58(9), Article 236, pp. 1–35, Feb 2026.
   https://doi.org/10.1145/3793661
   → *The strongest single citation for the marker layer:* a current, peer-reviewed CSUR
   survey of fiducial markers (ArUco and beyond), pose-estimation accuracy, and a design
   framework. Use to justify the choice of ArUco for floor-plane registration.

2. **Robust and Scalable Indoor Robot Localization Based on Fusion of Infrastructure
   Camera Feeds and On-Board Sensors.** Poornima J D, Raghu Krishnapuram, Mukunda
   Bharatheesha, Bharadwaj Amrutur, Suresh Sundaram. *AIR '23: 6th Int'l Conf. on Advances
   in Robotics*, Article 41, pp. 1–7, Nov 2023. https://doi.org/10.1145/3610419.3610460
   → Directly on point for **fusing multiple fixed (infrastructure) camera feeds** into a
   single pose estimate with spatially-varying uncertainty — the multi-camera fusion
   problem Drop Ceiling solves with its tracker.

3. **MoiréBoard: A Stable, Accurate and Low-cost Camera Tracking Method.**
   Chang Xiao, Changxi Zheng. *UIST '21: 34th ACM Symp. on User Interface Software and
   Technology*, pp. 881–893, Oct 2021. https://doi.org/10.1145/3472749.3474793
   → Optional: low-cost camera tracking/registration in an HCI venue; useful if you discuss
   the accuracy/cost trade-offs that motivated a marker-based rather than beacon-based setup.

4. **ArUcoTUI: Software Toolkit for Prototyping Tangible Interactions on Portable
   Flat-Panel Displays with OpenCV.** Rong-Hao Liang, Steven Houben, Krithik Ranjan,
   S. Sandra Bae, Peter Gyory, Ellen Yi-Luen Do, Clement Zheng. *TEI '26*, Article 58,
   pp. 1–8, Mar 2026. https://doi.org/10.1145/3731459.3779317
   → Optional/supporting: a recent ACM example of ArUco + OpenCV used as the interaction
   substrate, demonstrating the marker pipeline's standing as a current HCI tool.

---

## 2. Core algorithms & concepts of the self-tuning system

The autotuner is a budgeted online learner: a Thompson-sampling **multi-armed bandit**
chooses expression strategies, a gradient/score-driven tuner adjusts personality
parameters, all under an explicit exploration–exploitation trade-off and non-stationary
(time-of-day) conditions.

5. **A Tutorial on Multi-Armed Bandit Applications for Large Language Models.**
   Djallel Bouneffouf, Raphaël Féraud. *KDD '24: 30th ACM SIGKDD Conf. on Knowledge
   Discovery and Data Mining*, pp. 6412–6413, Aug 2024.
   https://doi.org/10.1145/3637528.3671440
   → A concise, citable ACM **overview of MAB methods** (incl. Thompson sampling) and the
   exploration/exploitation framing — good for grounding the bandit without a math detour.

6. **What You Reward Is What You Learn: Comparing Rewards for Online Speech Policy
   Optimization in Public HRI.** Sichao Song, Yuki Okafuji, Kaito Ariu, Amy Koike.
   *HRI '26: 21st ACM/IEEE Int'l Conf. on Human-Robot Interaction*, pp. 187–195, Mar 2026.
   https://doi.org/10.1145/3757279.3785589
   → *Unusually close analogue:* **online learning that adapts behaviour policy in a public
   space**, explicitly contrasting "fixed, hand-tuned parameters" with online adaptation to
   "non-stationary conditions" — almost exactly Drop Ceiling's argument for self-tuning, and
   it discusses reward design (the engagement-score question).

7. **Thompson Sampling with Unrestricted Delays.** Han Wu, Stefan Wager.
   *EC '22: 23rd ACM Conf. on Economics and Computation*, pp. 937–955, Jul 2022.
   https://doi.org/10.1145/3490486.3538376
   → Optional/technical: regret bounds for **Thompson sampling under delayed feedback** —
   relevant because engagement outcomes in the installation arrive with delay relative to
   the expression choice. Cite if you want a rigorous bandit anchor.

8. **A Behavioral Model for Exploration vs. Exploitation: Theoretical Framework and
   Experimental Evidence.** Jingying Ding, Yifan Feng, Ying Rong. *EC '25: 26th ACM Conf.
   on Economics and Computation*, p. 88, Jul 2025. https://doi.org/10.1145/3736252.3742497
   → Optional: frames the **exploration–exploitation trade-off** itself (the conceptual
   core of curiosity vs. mean-reversion in the tuner) via the bandit lens.

---

## 3. Augmented reality — framing the project (Spatial AR / projection-based)

For positioning Drop Ceiling as a **spatial/projected AR** work: digital response
registered to physical space, experienced without worn devices.

9. **DroneSAR: Extending Physical Spaces in Spatial Augmented Reality Using Projection on
   a Drone.** Rajkumar Darbar, Joan Sol Roo, Thibault Lainé, [+1]. *MUM '19: 18th Int'l
   Conf. on Mobile and Ubiquitous Multimedia*, Article 4, pp. 1–7, Nov 2019.
   https://doi.org/10.1145/3365610.3365631
   → Clean, citable **definition and use of SAR** ("transforms real-world objects into
   interactive displays … without the need to wear any special [gear]") — ideal for the
   sentence where you define the project's AR category.

10. **Diminishable Visual Markers on Fabricated Projection Object for Dynamic Spatial
    Augmented Reality.** Hirotaka Asayama, Daisuke Iwai, Kosuke Sato. *SIGGRAPH Asia '15
    Emerging Technologies*, Article 7, pp. 1–2, Nov 2015.
    https://doi.org/10.1145/2818466.2818477
    → Ties **SAR + fiducial markers** together in one reference — i.e. the same marker-based
    registration idea Drop Ceiling uses, framed explicitly within SAR.

11. **FamiliAR Feedback: Investigating Feedback Modality and Familiarity in Classroom
    Settings Using Spatial Augmented Reality.** Nick Wittig, Yannick Dohmen, Jonathan
    Liebers, [+3]. *MUM '25: 24th Int'l Conf. on Mobile and Ubiquitous Multimedia*,
    pp. 273–284, Nov 2025. https://doi.org/10.1145/3771882.3771883
    → A current SAR study showing the term and method are live in the recent ACM literature;
    useful as a "SAR remains an active research area" citation.

> **Foundational anchor to add (verify before citing):** Bimber & Raskar, *Spatial
> Augmented Reality: Merging Real and Virtual Worlds* (A K Peters, 2005) is the canonical
> SAR reference. It is a book, not ACM-hosted, so I did **not** verify it in DL; include it
> from the original if you want the field-defining citation alongside the ACM examples above.

---

## 4. Adjacent / framing — interactive light in public space, media architecture

Positions Drop Ceiling among permanent, camera-driven, public installations that respond
to and learn from passers-by — the cultural/architectural framing for ACADIA's
*Built Grounds* and *Machines that Care* subthemes.

12. **Engaging Passers-by with Rhythm: Applying Feedforward Learning to a Xylophonic Media
    Architecture Facade.** Binh Vinh Duc Nguyen, Jihae Han, [+3]. *CHI '23: 2023 CHI Conf.
    on Human Factors in Computing Systems*, Article 182, pp. 1–21, Apr 2023.
    https://doi.org/10.1145/3544548.3580761
    → *The closest sibling project found:* a **permanent media-architecture facade** that
    engages **passers-by** and applies **learning** — directly comparable on permanence,
    public-space engagement, and adaptive behaviour. Strong related-work citation.

13. **Towards Responsive Architecture that Mediates Place: Recommendations on How and When
    an Autonomously Moving Robotic Wall Should Adapt a Spatial Layout.** Binh Vinh Duc
    Nguyen, Jihae Han, Andrew Vande Moere. *PACMHCI* 6(CSCW2), Article 467, pp. 1–27,
    Nov 2022. https://doi.org/10.1145/3555568
    → On **responsive architecture** and *when* an autonomous system should adapt — speaks
    to Drop Ceiling's friction/restraint argument (acting deliberately, not constantly).

14. **Using Embodied Audio-Visual Interaction to Promote Social Encounters Around Large
    Media Façades.** Luke Hespanhol, Martin Tomitsch, Oliver Bown, [+1]. *DIS '14: 2014
    Conf. on Designing Interactive Systems*, pp. 945–954, Jun 2014.
    https://doi.org/10.1145/2598510.2598568
    → A **large-scale interactive light intervention in urban space** addressing the
    addressivity/engagement challenges of public media — good for the active/passive zone
    and "how do we invite engagement" discussion.

15. **Playing with the Spirit of a Place: Designing Urban Interactions with
    Hybrid-Resolution Media Facades.** Peggy Liu, Luke Hespanhol. *MAB '20: 5th Media
    Architecture Biennale Conference*, pp. 31–41, Oct 2021.
    https://doi.org/10.1145/3469410.3469414
    → Explicitly pairs **"responsive low-resolution LED lighting"** with high-res displays
    in urban play — supports the deliberately minimal 12-panel palette as a design stance,
    not a limitation.

16. **The Dual Skins of a Media Façade: Explicit and Implicit Interactions.**
    Claude Fortin, Kate Hennessy. *SIGGRAPH '15 Art Papers*, pp. 348–356, Jul 2015.
    https://doi.org/10.1145/2810177.2810181
    → On **implicit vs. explicit interaction** at a Canadian (Montréal) media façade —
    relevant to Drop Ceiling's mostly-implicit interaction model (presence/flow, not
    deliberate input) and its Toronto site.

---

## Quick-pick shortlist (if you only cite ~6)

| Need | Use |
|---|---|
| Markers / homography / fusion | **#1 (CSUR marker survey)** + **#2 (multi-camera fusion)** |
| Self-tuning algorithm | **#6 (online policy adaptation in public HRI)** + **#5 (MAB overview)** |
| AR framing | **#9 (SAR definition)** (+ Bimber & Raskar book from origin) |
| Public-space sibling work | **#12 (learning media facade, CHI '23)** |

*Notes: "[+N]" = additional authors truncated in ACM's list view; expand on the article
page before final citation. All DOIs verified resolvable on dl.acm.org. Pages/years copied
from ACM catalogue entries on 2026-05-31.*
