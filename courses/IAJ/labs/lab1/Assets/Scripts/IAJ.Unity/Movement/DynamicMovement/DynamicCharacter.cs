using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicCharacter : MonoBehaviour
    {
        [Header("Movement Settings")]
        [SerializeField]
        private float maxSpeed = 20.0f;
       [SerializeField]
        private float drag = 0.9f;
        //drag is a damping factor that reduces the character’s current velocity and rotation over time:
        //•	drag = 1.0f → no slowdown
        //•	drag = 0.9f → gradual slowdown
        //•	drag = 0.5f → strong slowdown
        //•	drag = 0.0f → velocity quickly disappears
        [SerializeField]
        private float maxAcceleration = 20.0f;

        public GameObject GameObject { get; protected set; }
        public KinematicData KinematicData { get; protected set; }
        private DynamicMovement movement;
        public DynamicMovement Movement
        {
            get { return this.movement; }
            set
            {
                this.movement = value;
                if (this.movement != null) this.movement.Character = this;
            }
        }
        public float Drag { get { return this.drag; } }
        public float MaxSpeed { get { return this.maxSpeed; } }
        public float MaxAcceleration { get { return this.maxAcceleration; } }

        public Vector3 Position { get { return this.KinematicData.Position; } }
        public float Orientation { get { return this.KinematicData.Orientation; } }
        public Vector3 Velocity { get { return this.KinematicData.velocity; } }

        private void Awake()
        {
            this.Initialize(this.gameObject);
        }

        public void Initialize(GameObject gameObject)
        {
            this.KinematicData = new KinematicData(new StaticData(gameObject.transform));
            this.GameObject = gameObject;
        }

        // Update is called once per frame
        public void UpdateMovement()
        {
            if (this.Movement != null)
            {
                MovementOutput output = this.Movement.GetMovement();

                if (output != null)
                {
                    Debug.DrawRay(this.GameObject.transform.position, output.linear, this.Movement.DebugColor);

                    this.KinematicData.Integrate(output, this.Drag, Time.deltaTime);
                    this.KinematicData.SetOrientationFromVelocity();
                    this.KinematicData.TrimMaxSpeed(this.MaxSpeed);
                }
            }
        }

    }
}
