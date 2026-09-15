using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public abstract class DynamicMovement : Movement
    {
        protected MovementOutput Output { get; set; }

        public DynamicCharacter Character { get; set; }
        public bool Active { get; set; }
        virtual public KinematicData Target { get; set; }

        public DynamicMovement()
        {
            this.Active = false;

        }
    }
}
