using UnityEngine;
using Assets.Scripts.IAJ.Unity.Util;

namespace Assets.Scripts.IAJ.Unity.Movement.KinematicMovement
{
    public class KinematicCharacter : MonoBehaviour
    {
        [Header("Movement Settings")]
        [SerializeField]
        private float maxSpeed = 10.0f;
        [SerializeField]
        private float maxRotation = 8 * MathConstants.MATH_PI;

        private KinematicMovement movement;

        public KinematicMovement Movement
        {
            get { return this.movement; }
            set
            {
                this.movement = value;
                if(this.movement != null) this.movement.Character = this.StaticData;
            }
        }

        public StaticData StaticData { get; protected set; }

        public GameObject GameObject { get; protected set; }

        public float MaxSpeed { get { return this.maxSpeed; } }

        public float MaxRotation { get { return this.maxRotation; } }

        private void Awake()
        {
            this.GameObject = this.gameObject;
            this.StaticData = new StaticData(this.transform);
        }

        public void Initialize(GameObject gameObject)
        {
            this.GameObject = gameObject;
            this.StaticData = new StaticData(gameObject.transform);
        }

        public void UpdateMovement()
        {
            if (this.Movement != null) 
            {
                MovementOutput output = this.Movement.GetMovement();

                if (output != null)
                {
                    this.StaticData.Integrate(output, Time.deltaTime);

                    if (!(this.Movement is KinematicWander))
                    {
                        this.StaticData.SetOrientationFromVelocity(output.linear);
                    }
                }
            }
        }
    }
}
